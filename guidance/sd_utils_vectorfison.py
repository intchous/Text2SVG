import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
import os
import random
import numpy as np
from os.path import isfile
from pathlib import Path

from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import CLIPTextModel, CLIPTokenizer, logging

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available

from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.loaders import AttnProcsLayers

import kornia.augmentation as K

# suppress partial model loading warning
logging.set_verbosity_error()


def seed_everything(seed):
    # set random seed everywhere
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)


def get_data_augs(cut_size, use_RandomResizedCrop=False):
    augmentations = []
    augmentations.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))

    if (use_RandomResizedCrop):
        # Random Resized Crop
        augmentations.append(K.RandomResizedCrop(
            (cut_size, cut_size), scale=(0.7, 1.0)))
    else:
        augmentations.append(K.RandomCrop(
            size=(cut_size, cut_size), pad_if_needed=True, padding_mode='reflect', p=1.0))

    return nn.Sequential(*augmentations)


def add_titles_to_images(viz_images, titles):

    to_pil = transforms.ToPILImage()
    images = [np.array(to_pil(img)) for img in viz_images]

    font_scale = 1.0
    font_color = (0, 0, 0)  # 黑色
    font_thickness = 2
    title_height = 60

    processed_images = []
    for i, image in enumerate(images):
        h, w, _ = image.shape
        title_area = np.ones((title_height, w, 3),
                             dtype=np.uint8) * 255
        image_with_title_area = np.vstack((image, title_area))

        if i < len(titles):
            text_size = cv2.getTextSize(
                titles[i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h + 30 + text_size[1] // 2
            cv2.putText(image_with_title_area, titles[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, font_color, font_thickness)

        processed_images.append(image_with_title_area)

    tensor_images = torch.stack([transforms.ToTensor()(
        Image.fromarray(img)) for img in processed_images])

    return tensor_images


class mSDSLoss(nn.Module):
    def __init__(self,  sd_version='svg', concept_n_pre="", concept_dir="", t_range=[0.05, 0.95], device="cuda", fp16=False, vram_O=False, hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == 'xl-base-1.0':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"

        elif self.sd_version == 'svg':
            # fine-tuned on icon dataset based on sd1.5
            concept_n = concept_n_pre + "_dreambooth_concept"
            model_key = os.path.join(concept_dir, concept_n)
        else:
            model_key = sd_version
            # raise ValueError(
            #     f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.precision_t)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
        else:
            pipe.to(device)

        pipe.safety_checker = lambda images, clip_input: (images, None)

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        # for convenience
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        self.alphas_sqrt = torch.sqrt(self.scheduler.alphas_cumprod).to(
            self.device, dtype=self.precision_t)
        self.sigmas_sqrt = torch.sqrt(
            1 - self.scheduler.alphas_cumprod).to(self.device, dtype=self.precision_t)

        pred_x0_total_timesteps = self.max_step - self.min_step + 1
        self.scheduler.set_timesteps(pred_x0_total_timesteps)

        self.base_prompt = "minimal flat 2d vector icon. lineal color. on a white background. trending on artstation; professional vector illustration; flat cartoon icon"

        self.base_negative_prompt = ""

        self.neg_embeds = self.get_text_embeds(self.base_negative_prompt)

        print(f'[INFO] loaded stable diffusion!')

    def set_t_range(self, t_range=[0.05, 0.95]):
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(prompt, padding='max_length',
                                max_length=self.tokenizer.model_max_length, return_tensors='pt', truncation=True)

        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def encode_text_posneg(self, prompt):
        # tokenizer and embed text
        text_embeddings = self.get_text_embeds(prompt)
        uncond_text_embeddings = self.neg_embeds
        text_embeddings = torch.cat([uncond_text_embeddings, text_embeddings])

        return text_embeddings

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = imgs * 2.0 - 1.0
        # imgs = (2 * imgs - 1).clamp(-1.0, 1.0)

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=10.0, as_latent=False, grad_scale=1.0, save_guidance_path=False):

        batch_size = pred_rgb.shape[0]

        # pred_rgb: BCHW
        if as_latent:
            latents = F.interpolate(
                pred_rgb, (64, 64), mode="bilinear", align_corners=False)
            # latents = latents * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(
                pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1,
                          (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # noise.shape [1, 4, 64, 64]
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
            # noise_pred.shape [2, 4, 64, 64]

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_pos - noise_pred_uncond)

        # ---------------------------------
        # from threestudio
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        # ---------------------------------

        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                eps = noise
                e_t = noise_pred
                alpha_t = self.alphas_sqrt[t, None, None, None]
                sigma_t = self.sigmas_sqrt[t, None, None, None]
                z_t = alpha_t * latents + sigma_t * eps
                pred_z0 = (z_t - sigma_t * e_t) / alpha_t

                min_val = grad.min()
                max_val = grad.max()

                normalized_grad = (grad - min_val) / (max_val - min_val)

                grad_vis_norm = (normalized_grad[:, :3, :, :]).clamp(0, 1)

                vis_grad = F.interpolate(grad_vis_norm, size=(
                    512, 512), mode='bilinear', align_corners=False)

                # ---------------------------------
                result_hopefully_less_noisy_image_z0 = self.decode_latents(
                    pred_z0.to(latents.type(self.precision_t)))

                # visualize noisier image [0,1]
                result_noisier_image = self.decode_latents(
                    latents_noisy.to(pred_z0).type(self.precision_t))

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                pred_rgb_512 = self.decode_latents(latents)
                viz_images = torch.cat(
                    [pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image_z0, vis_grad], dim=0)

                # ---------------------------------
                titles = ["t=" + str(t.item()),
                          'noisier_image', 'pred_z0', 'vis_grad']
                viz_images = add_titles_to_images(viz_images, titles)

        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * grad_scale * \
            F.mse_loss(latents, target, reduction="sum") / batch_size

        if (save_guidance_path):
            return loss, viz_images

        return loss

    def predict_noise0_diffuser(self, unet, latents_noisy, text_embeddings, t, guidance_scale=10.0, cross_attention_kwargs={}):
        batch_size = latents_noisy.shape[0]

        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2)

        tt = torch.cat([t] * 2)

        if guidance_scale == 1.:
            noise_pred = unet(
                latents_noisy, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=cross_attention_kwargs).sample

        else:
            # predict the noise residual
            noise_pred = unet(
                latent_model_input, tt, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
            # noise_pred.shape [2, 4, 64, 64]

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_pos - noise_pred_uncond)

        noise_pred = noise_pred.float()

        return noise_pred

    def sds_vsd_grad_diffuser(self, text_embeddings, pred_rgb, guidance_scale=10.0,  grad_scale=1.0, unet_phi=None, cfg_phi=1.0, generation_mode='sds', cross_attention_kwargs={'scale': 1.0}, as_latent=False, save_guidance_path=False):

        batch_size = pred_rgb.shape[0]

        # pred_rgb: BCHW
        if as_latent:
            latents = F.interpolate(
                pred_rgb, (64, 64), mode="bilinear", align_corners=False)
            # latents = latents * 2 - 1

        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(
                pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        self.latents = latents

        unet_cross_attention_kwargs = {'scale': 0} if (
            generation_mode == 'vsd') else {}

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1,
                          (latents.shape[0],), dtype=torch.long, device=self.device)

        # add noise
        noise = torch.randn_like(latents)  # noise.shape [1, 4, 64, 64]
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            noise_pred = self.predict_noise0_diffuser(unet=self.unet, latents_noisy=latents_noisy, text_embeddings=text_embeddings,
                                                      t=t, guidance_scale=guidance_scale, cross_attention_kwargs=unet_cross_attention_kwargs)

        if generation_mode == 'sds':
            # SDS
            grad = (noise_pred - noise)
            noise_pred_phi = noise

        elif generation_mode == 'vsd':
            with torch.no_grad():
                noise_pred_phi = self.predict_noise0_diffuser(
                    unet=unet_phi, latents_noisy=latents_noisy, text_embeddings=text_embeddings, t=t, guidance_scale=cfg_phi, cross_attention_kwargs=cross_attention_kwargs)

            # VSD
            grad = (noise_pred - noise_pred_phi.detach())

        # ---------------------------------
        # from threestudio
        w = (1. - self.alphas[t]).view(-1, 1, 1, 1)
        # w = self.sqrt_1m_alphas_cumprod[t]**2
        # grad = w * (noise_pred - noise)
        grad = w * grad
        # ---------------------------------
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                pred_latents = self.scheduler.step(
                    noise_pred.to('cpu'), t.to('cpu'), latents_noisy.to('cpu')).pred_original_sample.clone().detach().to(self.device)

                pred_z0 = pred_latents

                if generation_mode == 'vsd':
                    pred_latents_phi = self.scheduler.step(
                        noise_pred_phi.to('cpu'), t.to('cpu'), latents_noisy.to('cpu')).pred_original_sample.clone().detach().to(self.device)

                # ---------------------------------
                min_val = grad.min()
                max_val = grad.max()
                normalized_grad = (grad - min_val) / (max_val - min_val)

                # grad_vis_norm = (grad[:, :3, :, :] / 2 + 0.5).clamp(0, 1)
                grad_vis_norm = (normalized_grad[:, :3, :, :]).clamp(0, 1)

                vis_grad = F.interpolate(grad_vis_norm, size=(
                    512, 512), mode='bilinear', align_corners=False)

                # ---------------------------------
                result_hopefully_less_noisy_image_z0 = self.decode_latents(
                    pred_z0.to(latents.type(self.precision_t)))

                if generation_mode == 'vsd':
                    image_x0_phi = self.decode_latents(
                        pred_latents_phi.to(latents.type(self.precision_t)))

                # visualize noisier image [0,1]
                result_noisier_image = self.decode_latents(
                    latents_noisy.to(pred_z0).type(self.precision_t))

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                pred_rgb_512 = self.decode_latents(latents)

                # ---------------------------------
                if (generation_mode == 'vsd'):
                    viz_images = torch.cat(
                        [pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image_z0, image_x0_phi, vis_grad], dim=0)
                    titles = ["t=" + str(t.item()),
                              'noisier_image', 'pred_z0', 'image_x0_phi', 'vis_grad']

                else:
                    viz_images = torch.cat(
                        [pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image_z0, vis_grad], dim=0)

                    titles = ["t=" + str(t.item()),
                              'noisier_image', 'pred_z0', 'vis_grad']

                viz_images = add_titles_to_images(viz_images, titles)

        # -------------------------------------------
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * grad_scale * \
            F.mse_loss(latents, target, reduction="sum") / batch_size

        if (save_guidance_path):
            return loss, viz_images

        return loss

    def phi_vsd_grad_diffuser(self, unet_phi, pred_rgb, text_embeddings, cfg_phi=1., grad_scale=1, cross_attention_kwargs={'scale': 1.0}, as_latent=False):

        latents = self.latents

        t_phi = np.random.choice(
            list(range(self.num_train_timesteps)), 1, replace=True)[0]
        t_phi = torch.tensor([t_phi]).to(self.device)

        latents_phi = latents
        noise_phi = torch.randn_like(latents_phi)
        noisy_latents_phi = self.scheduler.add_noise(
            latents_phi, noise_phi, t_phi)

        latents_noisy = noisy_latents_phi.detach()

        # ---------------------------------
        loss_fn = nn.MSELoss()

        # ref to https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/sd_utils.py#L114

        # predict the noise residual with unet
        # clean_latents = self.scheduler.step(noise_phi, t_phi, latents_noisy).pred_original_sample

        noise_pred = self.predict_noise0_diffuser(unet=unet_phi, latents_noisy=latents_noisy, text_embeddings=text_embeddings,
                                                  t=t_phi, guidance_scale=cfg_phi, cross_attention_kwargs=cross_attention_kwargs)

        target = noise_phi
        loss = loss_fn(noise_pred, target)
        loss *= grad_scale

        return loss

    def extract_lora_diffusers(self):
        # ref: https://github.com/huggingface/diffusers/blob/4f14b363297cf8deac3e88a3bf31f59880ac8a96/examples/dreambooth/train_dreambooth_lora.py#L833

        # begin lora
        # Set correct lora layers
        unet_lora_attn_procs = {}
        for name, attn_processor in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith(
                "attn1.processor") else self.unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]

            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id]

            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = LoRAAttnProcessor

            unet_lora_attn_procs[name] = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            ).to(self.device)

        self.unet.set_attn_processor(unet_lora_attn_procs)
        unet_lora_layers = AttnProcsLayers(self.unet.attn_processors)

        # self.unet.requires_grad_(True)
        self.unet.requires_grad_(False)

        for param in unet_lora_layers.parameters():
            param.requires_grad_(True)

        # self.params_to_optimize = unet_lora_layers.parameters()
        # end lora

        return self.unet, unet_lora_layers

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn(
                (text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)[
                'prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        text_embeds = self.encode_text_posneg(
            prompt=prompts, negative_prompt=negative_prompts)

        # Text embeds -> img latents [1, 4, 64, 64]
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs
