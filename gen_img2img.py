import argparse
import shutil
import os
import torch
import requests
import torch
import PIL
from PIL import Image
from io import BytesIO
import random
import cairosvg

from transformers import CLIPImageProcessor, CLIPModel, CLIPFeatureExtractor
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DiffusionPipeline


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


def load_target_whitebg(fp, img_size=64):

    target = PIL.Image.open(fp)

    if target.size != (img_size, img_size):
        target = target.resize((img_size, img_size),
                               PIL.Image.Resampling.BICUBIC)

    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # target = np.array(target)
    return target


def load_img2img_pipe(use_pipeline_type=3, model_id_or_path="runwayml/stable-diffusion-v1-5", precision_t=torch.float32, device="cuda"):

    if (use_pipeline_type == 1):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id_or_path, torch_dtype=precision_t)

    elif (use_pipeline_type == 2):
        # mega
        pipe = DiffusionPipeline.from_pretrained(
            model_id_or_path, custom_pipeline="stable_diffusion_mega", torch_dtype=precision_t)

    elif (use_pipeline_type == 3):
        # lpw
        pipe = DiffusionPipeline.from_pretrained(
            model_id_or_path,
            custom_pipeline="lpw_stable_diffusion",
            torch_dtype=precision_t
        )

    else:
        # clip guided
        clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        # feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id)
        feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
        clip_model = CLIPModel.from_pretrained(
            clip_model_id, torch_dtype=precision_t)

        pipe = DiffusionPipeline.from_pretrained(
            model_id_or_path,
            # custom_pipeline="clip_guided_stable_diffusion",
            custom_pipeline="/home/zhangpeiying/research/diffusion-svg/svg-sds-vsd/svg-sds/diffusers/examples/community/clip_guided_stable_diffusion_img2img.py",
            clip_model=clip_model,
            feature_extractor=feature_extractor,
            torch_dtype=precision_t
        )

    pipe = pipe.to(device)
    # pipe.enable_attention_slicing()
    pipe.safety_checker = lambda images, clip_input: (images, None)

    return pipe


def get_img2img(prompt, pipe, use_pipeline_type=3, strength=0.4, guidance_scale=8.0, num_images_per_prompt=3, num_inference_steps=50):

    if (use_pipeline_type == 2 or use_pipeline_type == 3):
        images = pipe.img2img(prompt=prompt,
                              negative_prompt=base_negprompt,
                              num_images_per_prompt=num_images_per_prompt,
                              num_inference_steps=num_inference_steps,
                              image=init_image,
                              strength=strength, guidance_scale=guidance_scale).images

    elif (use_pipeline_type == 1):
        images = pipe(prompt=prompt,
                      negative_prompt=base_negprompt,
                      num_images_per_prompt=num_images_per_prompt,
                      num_inference_steps=num_inference_steps,
                      image=init_image,
                      strength=strength, guidance_scale=guidance_scale).images

    else:
        generator = torch.Generator(device="cuda").manual_seed(0)
        images = pipe(
            prompt=prompt,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            image=init_image,
            strength=strength,
            guidance_scale=7.5,
            clip_guidance_scale=100,
            num_cutouts=4,
            use_cutouts=False,
            # generator=generator,
        ).images

    return images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sidx", type=int, default=-1)
    parser.add_argument("--sz", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_images_per_prompt", type=int, default=3)

    args = parser.parse_args()

    h = args.sz
    w = args.sz

    # -------------------------------------
    device = "cuda"
    precision_t = torch.float32

    # 'hakurei/waifu-diffusion'
    # "runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1-base", "stabilityai/stable-diffusion-2-base"
    model_key = "runwayml/stable-diffusion-v1-5"
    model_id_or_path = model_key

    pipe_prefix_list = ["sdi2i", "mega", "lpw", "clipg"]
    use_pipeline_type = 3

    pipe = load_img2img_pipe(
        use_pipeline_type=use_pipeline_type, model_id_or_path=model_id_or_path, precision_t=precision_t, device=device)

    # -------------------------------------
    # (minimal flat 2d vector icon:1.3)
    prompt_appd = "(minimal flat 2d vector icon). (lineal color). (on a white background). (trending on artstation). (smooth)"

    # base_negprompt = get_negtive_prompt_text()
    base_negprompt = "(((text))), (shading), background, noise, dithering, gradient, detailed, out of frame, ugly, stripe, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry"
    # -------------------------------------

    # -------------------------------------
    cairo_img_fp = ""
    svg_fp = ""
    cairosvg.svg2png(url=svg_fp, write_to=cairo_img_fp,
                     output_width=w, output_height=h)
    init_image = load_target_whitebg(cairo_img_fp, img_size=512)

    prompt_descrip = "a ..."
    prompt = prompt_descrip + ". " + prompt_appd
    print("Prompt:", prompt)

    # ---------------------------------
    # 0.4, 0.45, 0.75
    strength = args.strength

    # 8.0, 7.5, 9.0
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_steps
    num_images_per_prompt = args.num_images_per_prompt

    # ---------------------------------
    images = get_img2img(prompt, pipe, use_pipeline_type=use_pipeline_type, strength=strength, guidance_scale=guidance_scale,
                         num_images_per_prompt=num_images_per_prompt, num_inference_steps=num_inference_steps)

    # for t_idx, tmp_img in enumerate(images):
    #     tmp_img.save(aft_svg_img_fp)
