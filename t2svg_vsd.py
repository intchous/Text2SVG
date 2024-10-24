import argparse
import os
import shutil
import subprocess
import zipfile

import cairosvg
import clip
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import PIL
import PIL.Image
import PIL.ImageDraw
import pydiffvg
import skimage
import skimage.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

import yaml
import types
from easydict import EasyDict as edict
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm, trange

from losses import laplacian_smoothing_loss
from optim_utils.util_prompts import get_negtive_prompt_text, load_samp_prompts
from guidance.sd_utils_vectorfison import mSDSLoss, seed_everything, get_data_augs

from deepsvg.my_svg_dataset_pts import Normalize, SVGDataset_nopadding
from deepsvg.model.config import _DefaultConfig
from deepsvg.model.model_pts_vae import SVGTransformer
from deepsvg.test_utils import RandomCoordInit, load_model2, recon_to_affine_pts, pts_to_pathObj, save_paths_svg, render_and_compose, generate_single_affine_parameters, get_experiment_id

CUDA_version = [s for s in subprocess.check_output(
    ["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)

os.environ["FFMPEG_BINARY"] = "ffmpeg"


pydiffvg.set_print_timing(False)
# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
device = torch.device("cuda")
pydiffvg.set_device(device)
gamma = 1.0
render = pydiffvg.RenderFunction.apply


class m_linear_decay_lrlambda_f(object):
    def __init__(self, remaining_steps, final_lr_ratio):
        self.remaining_steps = remaining_steps
        self.final_lr_ratio = final_lr_ratio

    def __call__(self, n):
        if n <= self.remaining_steps:
            return (1 - n / self.remaining_steps) * (1 - self.final_lr_ratio) + self.final_lr_ratio
        else:
            over_steps = n - self.remaining_steps
            over_ratio = over_steps / self.remaining_steps
            extra_decay = self.final_lr_ratio ** (over_ratio + 1)
            return extra_decay


def initialize_optimizers_and_schedulers(z_list, theta_list, s_list, tx_list, ty_list, color_list, args_num_iter, z_lr=0.1, color_lr=0.01, affine_lr_initial=0.18, theta_lr_initial=0.0006, s_lr_initial=0.0018, cur_step=0, use_affine_norm=False):

    final_lr_ratio = 0.4
    current_lr_factor = (1 - cur_step / args_num_iter) * \
        (1 - final_lr_ratio) + final_lr_ratio

    z_optimizer = optim.Adam(z_list, lr=z_lr * current_lr_factor)
    color_optimizer = optim.Adam(color_list, lr=color_lr * current_lr_factor)

    if (use_affine_norm):
        affine_lr_initial = 0.005

    final_lr_ratio = 0.16
    current_lr_factor = (1 - cur_step / args_num_iter) * \
        (1 - final_lr_ratio) + final_lr_ratio

    affine_optimizer = torch.optim.Adam([
        {'params': theta_list, 'lr': theta_lr_initial * current_lr_factor},
        {'params': s_list, 'lr': s_lr_initial * current_lr_factor},
        {'params': tx_list, 'lr': affine_lr_initial * current_lr_factor},
        {'params': ty_list, 'lr': affine_lr_initial * current_lr_factor}
    ])

    remaining_steps = args_num_iter - cur_step
    lr_lambda = m_linear_decay_lrlambda_f(remaining_steps, final_lr_ratio)
    scheduler_z = LambdaLR(
        z_optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    scheduler_affine = LambdaLR(
        affine_optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    scheduler_color = LambdaLR(
        color_optimizer, lr_lambda=lr_lambda, last_epoch=-1)

    return z_optimizer, color_optimizer, affine_optimizer, scheduler_z, scheduler_affine, scheduler_color


# ----------------------------------------------------
def get_img_from_list(z_list, theta_list, tx_list, ty_list, s_list, color_list, model, s_norm, w=224, h=224, svg_path_fp="", use_affine_norm=False, render_func=None, return_shapes=True):

    z_batch = torch.stack(z_list).to(device).squeeze(1)
    generated_data_batch = model(
        args_enc=None, args_dec=None, z=z_batch.unsqueeze(1).unsqueeze(2))
    generated_pts_batch = generated_data_batch["args_logits"]
    recon_data_output_batch = generated_pts_batch.squeeze(
        1)

    # ---------------------------------------------
    tmp_paths_list = []
    for _idx in range(len(z_list)):

        convert_points, convert_points_ini = recon_to_affine_pts(
            recon_data_output=recon_data_output_batch[_idx], theta=theta_list[_idx], tx=tx_list[_idx], ty=ty_list[_idx], s=s_list[_idx], s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

        optm_convert_path = pts_to_pathObj(convert_points)
        tmp_paths_list.append(optm_convert_path)

    if (return_shapes):
        recon_imgs, tmp_img_render, tp_shapes, tp_shape_groups = render_and_compose(
            tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp=svg_path_fp, render_func=render, return_shapes=return_shapes)
        return recon_imgs, tmp_img_render, tp_shapes, tp_shape_groups

    else:
        recon_imgs, tmp_img_render = render_and_compose(
            tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp=svg_path_fp, render_func=render_func, return_shapes=return_shapes)
        return recon_imgs, tmp_img_render


# ----------------------------------------------------
def reinitialize_paths(cur_shapes, cur_shape_groups, z_list, theta_list, s_list, tx_list, ty_list, color_list, pos_init_method=None, opacity_threshold=0.05, area_threshold=10, add_new=True):
    """
    reinitialize paths, also known as 'Reinitializing paths' in VectorFusion paper.

    Args:
        reinit_path: whether to reinitialize paths or not.
        opacity_threshold: Threshold of opacity.
        area_threshold: Threshold of the closed polygon area.
        fpath: The path to save the reinitialized SVG.
    """

    # re-init by opacity_threshold
    select_path_ids_by_opc = []
    if opacity_threshold != 0 and opacity_threshold is not None:
        def get_keys_below_threshold(my_dict, threshold):
            keys_below_threshold = [
                key for key, value in my_dict.items() if value < threshold]
            return keys_below_threshold

        opacity_record_ = {group.shape_ids.item(
        ): group.fill_color.data[-1].item() for group in cur_shape_groups}

        # print("-> opacity_record: ",[f"{k}: {v:.3f}" for k, v in opacity_record_.items()])
        select_path_ids_by_opc = get_keys_below_threshold(
            opacity_record_, opacity_threshold)
        # print("select_path_ids_by_opc: ", select_path_ids_by_opc)

    # ---------------------------------------------------
    cover_threshold = 0.9
    shape_polygons = []
    for shape in cur_shapes:
        polygon = Polygon(shape.points.detach().cpu().numpy())
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        shape_polygons.append(polygon)

    shape_areas = [polygon.area for polygon in shape_polygons]

    select_path_ids_by_area = []
    select_path_ids_by_cover = []
    for i, current_shape_polygon in enumerate(shape_polygons):
        current_area = shape_areas[i]

        if current_area < area_threshold:
            select_path_ids_by_area.append(i)
            continue

        intersections = [current_shape_polygon.intersection(shape_polygons[j])
                         for j in range(i + 1, len(shape_polygons))
                         if current_shape_polygon.intersects(shape_polygons[j])]
        covered_area = unary_union(intersections).area

        if covered_area * 1.0 / current_area > cover_threshold:
            select_path_ids_by_cover.append(i)

    # print("select_path_ids_by_area: ", select_path_ids_by_area)
    # print("select_path_ids_by_cover: ", select_path_ids_by_cover)
    # ---------------------------------------------------
    # re-init paths
    reinit_union = list(set(select_path_ids_by_opc +
                        select_path_ids_by_area + select_path_ids_by_cover))

    kept_shapes = []
    kept_zs = []
    kept_thetas = []
    kept_s = []
    kept_tx = []
    kept_ty = []
    kept_colors = []

    new_zs = []
    new_thetas = []
    new_s = []
    new_tx = []
    new_ty = []
    new_colors = []

    for shape_index, shape in enumerate(cur_shapes):
        if shape_index not in reinit_union:
            kept_shapes.append(shape)
            kept_colors.append(cur_shape_groups[shape_index].fill_color)
            kept_zs.append(z_list[shape_index])
            kept_thetas.append(theta_list[shape_index])
            kept_s.append(s_list[shape_index])
            kept_tx.append(tx_list[shape_index])
            kept_ty.append(ty_list[shape_index])

        else:
            center = np.array(pos_init_method())
            tmp_tx, tmp_ty, tmp_theta, tmp_s, tmp_fill_color_init = generate_single_affine_parameters(
                center=center, h=h, w=w, use_affine_norm=use_affine_norm, device=device)

            tmp_z = get_z_from_circle(absolute_base_dir=absolute_base_dir).to(
                device).requires_grad_(True)

            new_zs.append(tmp_z)
            new_thetas.append(tmp_theta)
            new_s.append(tmp_s)
            new_tx.append(tmp_tx)
            new_ty.append(tmp_ty)
            new_colors.append(tmp_fill_color_init)

    if (add_new):
        cur_z_list = kept_zs + new_zs
        cur_theta_list = kept_thetas + new_thetas
        cur_s_list = kept_s + new_s
        cur_tx_list = kept_tx + new_tx
        cur_ty_list = kept_ty + new_ty
        cur_color_list = kept_colors + new_colors
    else:
        cur_z_list = kept_zs
        cur_theta_list = kept_thetas
        cur_s_list = kept_s
        cur_tx_list = kept_tx
        cur_ty_list = kept_ty
        cur_color_list = kept_colors

    return cur_z_list, cur_theta_list, cur_s_list, cur_tx_list, cur_ty_list, cur_color_list


# ----------------------------------------------------
def get_z_from_circle(absolute_base_dir):
    rd_fp_list = ["circle_10.svg"]

    dataset_h = 224
    dataset_w = 224
    svg_data_img_dir = os.path.join(absolute_base_dir, "vae_dataset/")

    tmp_svg_dataset_test = SVGDataset_nopadding(
        directory=svg_data_img_dir, h=dataset_h, w=dataset_w, fixed_length=max_pts_len_thresh, file_list=rd_fp_list, img_dir=svg_data_img_dir, transform=Normalize(dataset_w, dataset_h), use_model_fusion=cfg.use_model_fusion)

    tmp_test_loader = DataLoader(
        tmp_svg_dataset_test, batch_size=1, shuffle=False)

    for i, batch_data in enumerate(tmp_test_loader):
        points_batch = batch_data["points"]
        filepaths = batch_data["filepaths"]
        path_imgs = batch_data["path_img"]
        path_imgs = path_imgs.to(device)

        bat_s, _, _, _ = batch_data["cubics"].shape
        cubics_batch_fl = batch_data["cubics"].view(bat_s, -1, 2)

        lengths = batch_data["lengths"]
        data_pts = cubics_batch_fl.to(device)

        data_input = data_pts.unsqueeze(1)

        output = model(args_enc=data_input,
                       args_dec=data_input, ref_img=path_imgs)

        z1 = model.latent_z.detach().cpu().clone()

    return z1.squeeze(0).squeeze(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Load VSD config from yaml file")
    parser.add_argument("--vsd_cfn", type=str, default="t2svg_vsd_cfg",
                        help="Path name of the VSD yaml config file")

    vsd_args = parser.parse_args()

    vsd_yaml_fn = vsd_args.vsd_cfn
    vsd_yaml_fp = os.path.join("./configs", vsd_yaml_fn + ".yaml")

    with open(vsd_yaml_fp, 'r') as f:
        vsd_cfg_data = yaml.safe_load(f)

    vsd_cfg_data = types.SimpleNamespace(**vsd_cfg_data)
    print("vsd_cfg_data: ", vsd_cfg_data)

    # ---------------------------------
    cfg = _DefaultConfig()
    yaml_fn = "vae_config_cmd_10"
    yaml_fp = os.path.join("./configs", yaml_fn + ".yaml")

    with open(yaml_fp, 'r') as f:
        config_data = yaml.safe_load(f)

    for key, value in config_data.items():
        setattr(cfg, key, value)

    cfg.img_latent_dim = int(cfg.d_img_model / 64.0)
    cfg.vq_edim = int(cfg.dim_z / cfg.vq_comb_num)

    # ---------------------------------------
    latent_dim = cfg.dim_z
    max_pts_len_thresh = cfg.max_pts_len_thresh
    h = vsd_cfg_data.h
    w = vsd_cfg_data.w

    # ---------------------------------------
    sd_version = vsd_cfg_data.sd_version
    generation_mode = vsd_cfg_data.generation_mode
    guidance_scale = vsd_cfg_data.guidance_scale

    z_lr = vsd_cfg_data.z_lr
    color_lr = vsd_cfg_data.color_lr
    theta_lr_initial = vsd_cfg_data.theta_lr_initial
    affine_lr_initial = vsd_cfg_data.affine_lr_initial
    s_lr_initial = vsd_cfg_data.s_lr_initial

    add_num_paths = vsd_cfg_data.add_num_paths
    NUM_AUGS = vsd_cfg_data.NUM_AUGS
    GRAD_ACC = vsd_cfg_data.GRAD_ACC
    dataset_name = vsd_cfg_data.dataset_name

    tmp_seed = vsd_cfg_data.seed
    if (tmp_seed != -1):
        seed_everything(tmp_seed)
    # ---------------------------------
    vecf_log_par = os.path.join("./t2svg_logs", str(h) + "_" +
                                generation_mode + "_" + sd_version + "_test_" + dataset_name + "/")

    vecf_log_dir = os.path.join(vecf_log_par, "afflr_" + str(affine_lr_initial) +
                                "_thetalr_" + str(theta_lr_initial) + "_slr_" + str(s_lr_initial) + "_numps_" + str(add_num_paths) + "_GACC_" + str(GRAD_ACC) + "/")

    # ---------------------------------
    absolute_base_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))) + "/"
    model_dir = absolute_base_dir + "vae_model/"

    model = SVGTransformer(cfg)
    model = model.to(device)

    color_black = torch.FloatTensor([0, 0, 0, 1]).to("cuda")
    eos_threshold = 0.08

    desc = "cmd_10"
    print("desc: ", desc)

    model_save_dir = os.path.join(model_dir, desc)
    model_fp = os.path.join(model_save_dir, "best.pth")

    # load pretrained model
    load_model2(model_fp, model)
    model.eval()

    use_affine_norm = False
    s_norm = Normalize(w, h)
    # clip_loss_func = ClipLoss({})

    # ---------------------------------
    # CLIP loss type, might improve the results
    # @param ["spherical", "cosine"] {type: "string"}
    clip_loss_type = "cosine"
    # @param ["ViT-B/16", "ViT-B/32", "RN50", "RN50x4"]
    # [ViT-L/14, ViT-H/14, ViT-g/14]
    clip_version = "ViT-B/16"
    # ---------------------------------
    ini_svg_pre = "random"
    samp_prompts = load_samp_prompts(dataset_name=dataset_name, do_extend=True)

    diff_wh = 512
    if ("xl" in sd_version):
        diff_wh = 1024

    if (h < diff_wh):
        data_augs = get_data_augs(w)
    else:
        data_augs = get_data_augs(diff_wh)

    # -------------------------------------------------
    base_negprompt = get_negtive_prompt_text()
    description_start = "a clipart of "
    prompt_appd = "minimal flat 2d vector icon. lineal color. on a white background. trending on artstation."
    # --------------------------------
    cur_exid = str(get_experiment_id())
    for prompt_descrip in samp_prompts:
        svg_res_dir = os.path.join(
            vecf_log_dir, desc, prompt_descrip, cur_exid + "_gs" + str(guidance_scale))
        os.makedirs(svg_res_dir, exist_ok=True)

        # --------------------------------------------------
        sv_dir = os.path.join(svg_res_dir, "vecfu_process")
        os.makedirs(sv_dir, exist_ok=True)

        vis_pred_dir = os.path.join(svg_res_dir, "vis_pred")
        os.makedirs(vis_pred_dir, exist_ok=True)

        # --------------------------------
        description_end = " " + prompt_descrip + ", "
        prompt = prompt_descrip + " " + prompt_appd
        print("prompt: ", prompt)

        def args(): return None

        args.num_iter = vsd_cfg_data.num_iter

        guidance = mSDSLoss(sd_version=sd_version, device=device)
        text_embeddings = guidance.encode_text_posneg(prompt)
        args.num_paths = add_num_paths
        num_paths = args.num_paths

        # ----------------------------------------------------------------
        pos_init_method = RandomCoordInit(
            canvas_size=[h, w], edge_margin_ratio=0.05)

        inis_centers = []
        tx_list = []
        ty_list = []
        theta_list = []
        s_list = []
        color_list = []
        for i_p in range(num_paths):
            center = np.array(pos_init_method())
            inis_centers.append(center)

            tmp_tx, tmp_ty, tmp_theta, tmp_s, tmp_fill_color_init = generate_single_affine_parameters(
                center=center, h=h, w=w, use_affine_norm=use_affine_norm, device=device)

            tx_list.append(tmp_tx)
            ty_list.append(tmp_ty)
            theta_list.append(tmp_theta)
            s_list.append(tmp_s)
            color_list.append(tmp_fill_color_init)

        z_list = [get_z_from_circle(absolute_base_dir=absolute_base_dir).to(device).requires_grad_(True)
                  for _ in range(num_paths)]

        z_optimizer, color_optimizer, affine_optimizer, scheduler_z, scheduler_affine, scheduler_color = initialize_optimizers_and_schedulers(
            z_list=z_list, theta_list=theta_list, s_list=s_list, tx_list=tx_list, ty_list=ty_list, color_list=color_list, args_num_iter=args.num_iter, z_lr=z_lr, color_lr=color_lr, affine_lr_initial=affine_lr_initial, theta_lr_initial=theta_lr_initial, s_lr_initial=s_lr_initial, cur_step=0, use_affine_norm=use_affine_norm)

        # ---------------------------------------------------------------
        phi_model = 'lora'
        phi_optimizer = None
        phi_lr = 0.0001
        phi_update_step = 1

        if generation_mode == 'vsd':

            vae_phi = guidance.vae
            # unet_phi is the same instance as unet that has been modified in-place
            unet_phi, unet_lora_layers = guidance.extract_lora_diffusers()

            phi_params = list(unet_lora_layers.parameters())

            phi_optimizer = torch.optim.AdamW(
                [{"params": phi_params, "lr": phi_lr}], lr=phi_lr)
            print(
                f'number of trainable parameters of phi model in optimizer: {sum(p.numel() for p in phi_params if p.requires_grad)}')

        elif generation_mode == 'sds':
            unet_phi = None
            vae_phi = guidance.vae

        # ---------------------------------------------------------------
        best_loss = float('inf')
        best_iter = 0
        use_smooth_loss = False
        para_bg = torch.tensor(
            [1., 1., 1.], requires_grad=False, device=device)

        ini_cpdl_weight = 0.0001
        final_cpdl_weight = 0.00001
        cpdl_weight_decrement = (
            ini_cpdl_weight - final_cpdl_weight) / args.num_iter

        # Run the main optimization loop
        for t in trange(args.num_iter):

            z_optimizer.zero_grad()
            affine_optimizer.zero_grad()
            color_optimizer.zero_grad()

            for _ in range(GRAD_ACC):
                z_batch = torch.stack(z_list).to(device).squeeze(1)
                generated_data_batch = model(
                    args_enc=None, args_dec=None, z=z_batch.unsqueeze(1).unsqueeze(2))
                generated_pts_batch = generated_data_batch["args_logits"]
                recon_data_output_batch = generated_pts_batch.squeeze(
                    1)

                # -------------------------------------------
                tmp_paths_list = []
                tmp_new_points_list = []

                for _idx in range(len(z_list)):
                    convert_points, convert_points_ini = recon_to_affine_pts(
                        recon_data_output=recon_data_output_batch[_idx], theta=theta_list[_idx], tx=tx_list[_idx], ty=ty_list[_idx], s=s_list[_idx], s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

                    optm_convert_path = pts_to_pathObj(convert_points)

                    tmp_paths_list.append(optm_convert_path)
                    tmp_new_points_list.append(
                        optm_convert_path.points)

                # -------------------------------------------
                recon_imgs, img = render_and_compose(
                    tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp="", render_func=render)

                if t % 500 == 0:
                    tmp_svg_fp = os.path.join(sv_dir, "iter_{}.svg".format(t))
                    save_paths_svg(
                        path_list=tmp_paths_list, fill_color_list=color_list, svg_path_fp=tmp_svg_fp, canvas_height=h, canvas_width=w)

                tmp_imgs = recon_imgs.repeat(NUM_AUGS, 1, 1, 1)
                im_batch = data_augs.forward(tmp_imgs)

                m_sdloss = guidance.sds_vsd_grad_diffuser(text_embeddings=text_embeddings, pred_rgb=im_batch, guidance_scale=guidance_scale,  grad_scale=1.0,
                                                          unet_phi=unet_phi, cfg_phi=1.0, generation_mode=generation_mode, cross_attention_kwargs={'scale': 1.0}, as_latent=False, save_guidance_path=False)
                # -----------------------------------------------
                m_smoothness_loss = 0.0
                # 0.1, 5e-1, 1e-2
                smoothing_loss_weight = 2e-1

                if (use_smooth_loss):
                    for idx_path in range(len(tmp_new_points_list)):
                        cur_path_pts = tmp_new_points_list[idx_path]
                        cur_path_pts = cur_path_pts.to("cuda")

                        cur_path_m_smoothness_loss = laplacian_smoothing_loss(
                            cur_path_pts)

                        m_smoothness_loss += cur_path_m_smoothness_loss

                    m_smoothness_loss = m_smoothness_loss / \
                        len(tmp_new_points_list) * smoothing_loss_weight

                # ------------------------------------------------
                loss = m_sdloss + m_smoothness_loss

                if (best_loss > loss.item()):
                    best_loss = loss.item()
                    best_iter = t
                    tmp_svg_fp = os.path.join(sv_dir, "best.svg")

                    save_paths_svg(
                        path_list=tmp_paths_list, fill_color_list=color_list, svg_path_fp=tmp_svg_fp, canvas_height=h, canvas_width=w)

                loss.backward()

            # Take a gradient descent step.
            color_optimizer.step()
            affine_optimizer.step()
            z_optimizer.step()

            scheduler_color.step()
            scheduler_affine.step()
            scheduler_z.step()

            for color in color_list:
                color.data.clamp_(0.0, 1.0)

            if t % 200 == 0:
                print("loss:", loss.item())
                print("iteration:", t)

                current_lr_z = scheduler_z.get_last_lr()[0]
                current_lr_color = scheduler_color.get_last_lr()[0]
                print(f"Z Optimizer LR: {current_lr_z}")
                # print(f"Color Optimizer LR: {current_lr_color}")

                print("Current Learning Rates for affine Optimizer:")
                for i, lr in enumerate(scheduler_affine.get_last_lr()):
                    print(f"Param Group {i}: LR = {lr}")
                # ------------------------------------------------

            if ((t > 0 and (t % 60 == 0) and (t < args.num_iter - 800))):

                _, _, tp_shapes, tp_shape_groups = get_img_from_list(z_list=z_list, theta_list=theta_list, tx_list=tx_list, ty_list=ty_list, s_list=s_list,
                                                                     color_list=color_list, model=model, s_norm=s_norm, w=w, h=h, svg_path_fp="", use_affine_norm=use_affine_norm, render_func=render, return_shapes=True)

                if ((t == args.num_iter - 2)):
                    add_new_flg = False
                else:
                    add_new_flg = True

                # if (h == 224):
                #     area_threshold = 10
                # else:
                #     area_threshold = 25
                area_threshold = 5

                z_list, theta_list, s_list, tx_list, ty_list, color_list = reinitialize_paths(
                    cur_shapes=tp_shapes, cur_shape_groups=tp_shape_groups, z_list=z_list, theta_list=theta_list, tx_list=tx_list, ty_list=ty_list, s_list=s_list, color_list=color_list, pos_init_method=pos_init_method, opacity_threshold=0.05, area_threshold=area_threshold, add_new=add_new_flg)

                z_optimizer, color_optimizer, affine_optimizer, scheduler_z, scheduler_affine, scheduler_color = initialize_optimizers_and_schedulers(
                    z_list=z_list, theta_list=theta_list, s_list=s_list, tx_list=tx_list, ty_list=ty_list, color_list=color_list, args_num_iter=args.num_iter, z_lr=z_lr, color_lr=color_lr, affine_lr_initial=affine_lr_initial, theta_lr_initial=theta_lr_initial, s_lr_initial=s_lr_initial, cur_step=t, use_affine_norm=use_affine_norm)

                current_lr_z = scheduler_z.get_last_lr()[0]
                current_lr_affine = scheduler_affine.get_last_lr()[0]

            # ------------------------------------------------
            torch.cuda.empty_cache()

            ######## Do the gradient for unet_phi #########
            if generation_mode == 'vsd':
                # update the unet (phi) model
                for _ in range(phi_update_step):
                    phi_optimizer.zero_grad()

                    loss_phi = guidance.phi_vsd_grad_diffuser(
                        unet_phi=unet_phi, pred_rgb=im_batch, text_embeddings=text_embeddings, cfg_phi=1.0, grad_scale=1.0, cross_attention_kwargs={'scale': 1.0}, as_latent=False)

                    loss_phi.backward()
                    phi_optimizer.step()

            # ------------------------------------------------

        filename = os.path.join(
            svg_res_dir, ini_svg_pre + "_" + prompt_descrip + "_optm.svg")
        save_paths_svg(path_list=tmp_paths_list, fill_color_list=color_list,
                       svg_path_fp=filename, canvas_height=h, canvas_width=w)

        filename = os.path.join(
            svg_res_dir, ini_svg_pre + "_" + prompt_descrip + "_optm_best.svg")
        # copy from best.svg
        shutil.copyfile(tmp_svg_fp, filename)

        # -----------------------------------------------------
        tensor_sv_fp = os.path.join(
            svg_res_dir, ini_svg_pre + "_" + prompt_descrip + "_optm_tensors_lists.pt")
        torch.save({'z_list': z_list, 'tx_list': tx_list, 'ty_list': ty_list,
                    'theta_list': theta_list, 's_list': s_list, 'color_list': color_list}, tensor_sv_fp)
