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

from .prompt_utils import get_negtive_prompt_text, load_samp_prompts

from .guidance.sd_utils_vectorfison import mSDSLoss, seed_everything, get_data_augs

from notebooks.test_ini_path import RandomCoordInit, generate_single_affine_parameters

from deepsvg.live_utils import get_experiment_id

from deepsvg.my_svg_dataset_pts import Normalize, SVGDataset_nopadding

from deepsvg.model.config import _DefaultConfig
from deepsvg.model.model_pts_vae import SVGTransformer

from losses import laplacian_smoothing_loss

from deepsvg.test_utils import load_model2, recon_to_affine_pts, pts_to_pathObj, save_paths_svg, render_and_compose

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


# ----------------------------------------------------


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
    # print("current_lr_factor: ", current_lr_factor)

    z_optimizer = optim.Adam(z_list, lr=z_lr * current_lr_factor)

    final_lr_ratio = 0.4
    current_lr_factor = (1 - cur_step / args_num_iter) * \
        (1 - final_lr_ratio) + final_lr_ratio

    # color_lr = 0.01
    color_optimizer = optim.Adam(
        color_list, lr=color_lr * current_lr_factor)

    if (use_affine_norm):
        # 1.0 / h
        affine_lr_initial = 0.005

        # 1.0
        theta_lr_factor = 1.5
        s_lr_factor = 2.0
    else:

        theta_lr_factor = 0.0006

        s_lr_factor = 0.0018

    final_lr_ratio = 0.16
    current_lr_factor = (1 - cur_step / args_num_iter) * \
        (1 - final_lr_ratio) + final_lr_ratio

    affine_optimizer = torch.optim.Adam([
        {'params': theta_list, 'lr': theta_lr_initial * current_lr_factor},
        {'params': s_list, 'lr': s_lr_initial * current_lr_factor},
        {'params': tx_list, 'lr': affine_lr_initial * current_lr_factor},
        {'params': ty_list, 'lr': affine_lr_initial * current_lr_factor}
    ])

    # 创建学习率调度器，从当前学习率开始递减
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

    # ---------------------------------------------
    # z_list[0].shape:  torch.Size([1, 24])
    z_batch = torch.stack(z_list).to(device).squeeze(1)
    # 使用模型生成点序列（批处理）
    generated_data_batch = model(
        args_enc=None, args_dec=None, z=z_batch.unsqueeze(1).unsqueeze(2))
    generated_pts_batch = generated_data_batch["args_logits"]
    recon_data_output_batch = generated_pts_batch.squeeze(
        1)
    # recon_data_output_batch.shape:  torch.Size([60, 32, 2])
    # ---------------------------------------------

    # ---------------------------------------------

    tmp_paths_list = []
    for _idx in range(len(z_list)):

        convert_points, convert_points_ini = recon_to_affine_pts(
            recon_data_output=recon_data_output_batch[_idx], theta=theta_list[_idx], tx=tx_list[_idx], ty=ty_list[_idx], s=s_list[_idx], s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

        # 应用仿射变换生成路径对象
        optm_convert_path = pts_to_pathObj(convert_points)
        # convert_ini_path = pts_to_pathObj(convert_points_ini)

        tmp_paths_list.append(optm_convert_path)

    if (return_shapes):
        # 使用所有变换后的路径和对应颜色渲染图像
        recon_imgs, tmp_img_render, tp_shapes, tp_shape_groups = render_and_compose(
            tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp=svg_path_fp, render_func=render, return_shapes=return_shapes)
        return recon_imgs, tmp_img_render, tp_shapes, tp_shape_groups

    else:
        recon_imgs, tmp_img_render = render_and_compose(
            tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp=svg_path_fp, render_func=render_func, return_shapes=return_shapes)
        return recon_imgs, tmp_img_render


# ----------------------------------------------------


def get_img_from_records(z_list_record, theta_list_record, tx_list_record, ty_list_record, s_list_record, color_list_record, model, s_norm, w=224, h=224, svg_path_fp="", use_affine_norm=False, render_func=None, return_shapes=True):

    tmp_paths_list = []
    tmp_colors_list = []

    # ---------------------------------------------
    # 合并所有记录中的z_list
    z_batch_all_records = torch.cat([torch.stack(z_list).squeeze(
        1) for z_list in z_list_record], dim=0).to(device)
    # z_list[0].shape:  torch.Size([1, 24])

    generated_data_batch_all = model(
        args_enc=None, args_dec=None, z=z_batch_all_records.unsqueeze(1).unsqueeze(2))
    generated_pts_batch_all = generated_data_batch_all["args_logits"]
    recon_data_output_batch_all = generated_pts_batch_all.squeeze(1)
    # recon_data_output_batch_all.shape:  torch.Size([60, 32, 2])

    # ---------------------------------------------

    start_idx = 0

    num_records = len(z_list_record)
    for record_idx in range(num_records):
        z_list = z_list_record[record_idx]
        theta_list = theta_list_record[record_idx]
        tx_list = tx_list_record[record_idx]
        ty_list = ty_list_record[record_idx]
        s_list = s_list_record[record_idx]
        color_list = color_list_record[record_idx]

        for idx in range(len(z_list)):

            recon_data_output = recon_data_output_batch_all[start_idx + idx]

            convert_points, _ = recon_to_affine_pts(
                recon_data_output=recon_data_output, theta=theta_list[idx], tx=tx_list[idx], ty=ty_list[idx], s=s_list[idx], s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

            optm_convert_path = pts_to_pathObj(convert_points)
            tmp_paths_list.append(optm_convert_path)

            tmp_colors_list.append(color_list[idx])

        start_idx += len(z_list)

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

        # print("-> opacity_record: ", opacity_record_)
        print("-> opacity_record: ",
              [f"{k}: {v:.3f}" for k, v in opacity_record_.items()])
        select_path_ids_by_opc = get_keys_below_threshold(
            opacity_record_, opacity_threshold)
        print("select_path_ids_by_opc: ", select_path_ids_by_opc)

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

        # 检测被遮挡的形状
        intersections = [current_shape_polygon.intersection(shape_polygons[j])
                         for j in range(i + 1, len(shape_polygons))
                         if current_shape_polygon.intersects(shape_polygons[j])]
        covered_area = unary_union(intersections).area

        if covered_area * 1.0 / current_area > cover_threshold:
            select_path_ids_by_cover.append(i)

    print("select_path_ids_by_area: ", select_path_ids_by_area)
    print("select_path_ids_by_cover: ", select_path_ids_by_cover)
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

    new_initialized_shapes = []
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
    rd_fp_list = ["2719851_cubic_3_r0.svg"]

    dataset_h = 224
    dataset_w = 224

    tmp_svg_dataset_test = SVGDataset_nopadding(
        directory=absolute_base_dir + "dataset/ini_svgs_470510_cubic_single_fit/", h=dataset_h, w=dataset_w, fixed_length=max_pts_len_thresh, file_list=rd_fp_list, img_dir=svg_data_img_dir, transform=Normalize(dataset_w, dataset_h), use_model_fusion=cfg.use_model_fusion)

    tmp_test_loader = DataLoader(
        tmp_svg_dataset_test, batch_size=1, shuffle=False)

    for i, batch_data in enumerate(tmp_test_loader):
        points_batch = batch_data["points"]
        filepaths = batch_data["filepaths"]
        path_imgs = batch_data["path_img"]
        path_imgs = path_imgs.to(device)

        bat_s, _, _, _ = batch_data["cubics"].shape
        # cubics_batch = batch_data["cubics"]
        cubics_batch_fl = batch_data["cubics"].view(bat_s, -1, 2)

        lengths = batch_data["lengths"]
        cubic_lengths = lengths // 3

        # data_pts = points_batch.to(device)
        data_pts = cubics_batch_fl.to(device)

        # 增加第二个维度
        data_input = data_pts.unsqueeze(1)
        # data_input.shape:  torch.Size([1, 1, 62, 2])

        output = model(args_enc=data_input,
                       args_dec=data_input, ref_img=path_imgs)

        z1 = model.latent_z.detach().cpu().clone()

    return z1.squeeze(0).squeeze(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Load VSD config from yaml file")
    parser.add_argument("--vsd_cfn", type=str, default="vecf_vsd_cfg",
                        help="Path name of the VSD yaml config file")

    vsd_args = parser.parse_args()

    vsd_yaml_fn = vsd_args.vsd_cfn
    vsd_yaml_fp = os.path.join(
        "./deepsvg/config_files/", vsd_yaml_fn + ".yaml")

    with open(vsd_yaml_fp, 'r') as f:
        vsd_cfg_data = yaml.safe_load(f)

    vsd_cfg_data = types.SimpleNamespace(**vsd_cfg_data)
    print("vsd_cfg_data: ", vsd_cfg_data)

    # ---------------------------------
    cfg = _DefaultConfig()

    yaml_fn = "vectorfusion_transformer_v5_nopadding"

    yaml_fp = os.path.join("./deepsvg/config_files/", yaml_fn + ".yaml")

    with open(yaml_fp, 'r') as f:
        config_data = yaml.safe_load(f)

    for key, value in config_data.items():
        setattr(cfg, key, value)

    # 计算并更新cfg
    cfg.img_latent_dim = int(cfg.d_img_model / 64.0)
    cfg.vq_edim = int(cfg.dim_z / cfg.vq_comb_num)

    # ---------------------------------------
    input_dim = cfg.n_args
    output_dim = cfg.n_args
    hidden_dim = cfg.d_model
    latent_dim = cfg.dim_z
    max_pts_len_thresh = cfg.max_pts_len_thresh
    kl_coe = cfg.kl_coe

    batch_size = cfg.batch_size

    log_interval = 20
    validate_interval = 4

    h = vsd_cfg_data.h
    w = vsd_cfg_data.w

    # ---------------------------------
    # 1.5, 2.0, 2.1
    sd_version = vsd_cfg_data.sd_version

    # vsd, sds
    generation_mode = vsd_cfg_data.generation_mode

    # SD guidance scale
    guidance_scale = vsd_cfg_data.guidance_scale

    z_lr = vsd_cfg_data.z_lr
    color_lr = vsd_cfg_data.color_lr
    theta_lr_initial = vsd_cfg_data.theta_lr_initial
    affine_lr_initial = vsd_cfg_data.affine_lr_initial
    s_lr_initial = vsd_cfg_data.s_lr_initial

    # 32, 64, 90, 100, 128, 64 * 4, 64 * 8
    add_num_paths = vsd_cfg_data.add_num_paths

    # 1, 4
    NUM_AUGS = vsd_cfg_data.NUM_AUGS

    # 1, 2, 4
    GRAD_ACC = vsd_cfg_data.GRAD_ACC

    # "pig", animal, animatesvg, vectorfusion
    dataset_name = vsd_cfg_data.dataset_name

    # 42, 1024
    tmp_seed = vsd_cfg_data.seed
    if (tmp_seed != -1):
        seed_everything(tmp_seed)
    # ---------------------------------

    vecf_log_par = "./transformer_vae_logs/vecfusion_smlr_" + \
        str(h) + "_" + generation_mode + "_" + \
        sd_version + "_test_" + dataset_name + "/"

    vecf_log_dir = os.path.join(vecf_log_par, "afflr_" + str(affine_lr_initial) +
                                "_thetalr_" + str(theta_lr_initial) + "_slr_" + str(s_lr_initial) + "_numps_" + str(add_num_paths) + "_GACC_" + str(GRAD_ACC) + "/")

    # ---------------------------------

    signature = "ini_svgs_470510"

    absolute_base_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))) + "/"

    svg_data_dir = os.path.join(
        absolute_base_dir + "dataset/", signature + "_cubic_single_fit/")
    svg_data_img_dir = os.path.join(
        absolute_base_dir + "dataset/", signature + "_cubic_single_img/")

    model_dir = absolute_base_dir + "transformer_vae_logs/models/"

    model = SVGTransformer(cfg)
    model = model.to(device)

    each_cubic_sample_num = 6
    color_black = torch.FloatTensor([0, 0, 0, 1]).to("cuda")
    # 0.16, 0.18
    # eos_threshold = 0.18
    eos_threshold = 0.08

    desc = "naive_vae_transformer_v1-5-7_" + "dataset-" + signature + "_" + "kl-" + \
        str(kl_coe) + "_" + "hd-" + str(hidden_dim) + "_" + "ld-" + str(latent_dim) + \
        "_" + "avg-" + str(cfg.avg_path_zdim) + "_" + "vae-" + \
        str(cfg.use_vae) + "_" + "sigm-" + str(cfg.use_sigmoid) + \
        "_" + "usemf-" + str(cfg.use_model_fusion) + "_" + \
        "losswl1-" + str(cfg.loss_w_l1) + "_" + "mce-" + \
        str(cfg.ModifiedConstEmbedding)

    print("desc: ", desc)

    model_save_dir = os.path.join(model_dir, desc)
    model_fp = os.path.join(model_save_dir, "best.pth")
    # model_fp = os.path.join(model_save_dir, "epoch_12.pth")

    # load pretrained model
    load_model2(model_fp, model)
    model.eval()

    s_norm = Normalize(w, h)
    # clip_loss_func = ClipLoss({})

    use_affine_norm = False

    # ---------------------------------

    # CLIP loss type, might improve the results
    # @param ["spherical", "cosine"] {type: "string"}
    clip_loss_type = "cosine"
    # @param ["ViT-B/16", "ViT-B/32", "RN50", "RN50x4"]
    # [ViT-L/14, ViT-H/14, ViT-g/14]
    clip_version = "ViT-B/16"

    svg_pre_prompt_dict = {
        "02866_peopwhole_18": "holding coffee",
    }

    svg_pre_name_dict = {
        "01729B_peopact_20": "a people character",
        "03670_animal_7": "a cattle",
    }

    # ---------------------------------
    samp_prompts = load_samp_prompts(dataset_name=dataset_name, do_extend=True)

    # ---------------------------------------------

    concept_dir = os.path.join(vecf_log_dir, "dreambooth_concept/")
    ini_svg_pre = "random"
    concept_n_pre = "02866_peopwhole_18"

    prompt_name = svg_pre_name_dict[concept_n_pre]

    diff_wh = 512
    if ("xl" in sd_version):
        diff_wh = 1024

    if (h < diff_wh):
        data_augs = get_data_augs(w)
    else:
        data_augs = get_data_augs(diff_wh)

    # -------------------------------------------------
    description_start = "a clipart of "
    prompt_appd = "minimal flat 2d vector icon. lineal color. on a white background. trending on artstation."

    base_negprompt = get_negtive_prompt_text()

    # --------------------------------

    cur_exid = str(get_experiment_id())

    for prompt_descrip in samp_prompts:

        svg_res_dir = os.path.join(
            vecf_log_dir, desc, prompt_descrip, cur_exid + "_gs" + str(guidance_scale))
        os.makedirs(svg_res_dir, exist_ok=True)

        # --------------------------------------------------
        sv_dir = os.path.join(svg_res_dir, "vecfu_process")
        os.makedirs(sv_dir, exist_ok=True)

        reinit_dir = os.path.join(svg_res_dir, "reinit_record")
        os.makedirs(reinit_dir, exist_ok=True)

        vis_pred_dir = os.path.join(svg_res_dir, "vis_pred")
        os.makedirs(vis_pred_dir, exist_ok=True)

        # --------------------------------

        description_end = " " + prompt_descrip + ", "
        prompt = prompt_descrip + " " + prompt_appd
        print("prompt: ", prompt)

        def args(): return None

        # 400, 500, 600, 800, 1000, 2000
        args.num_iter = vsd_cfg_data.num_iter

        guidance = mSDSLoss(sd_version=sd_version, concept_n_pre=concept_n_pre,
                            concept_dir=concept_dir, device=device)

        # --------------------------------------------------
        # 40, 60, 90
        BASE_STEPS = 60
        # 7.5, 8, 9
        BASE_SCALE = 9

        num_images_per_prompt = 3
        images = guidance.pipe(prompt,
                               negative_prompt=base_negprompt,
                               num_images_per_prompt=num_images_per_prompt,
                               num_inference_steps=BASE_STEPS,
                               guidance_scale=BASE_SCALE).images

        for g_idx, g_img in enumerate(images):
            tmp_fp = os.path.join(
                svg_res_dir, f"{ini_svg_pre}_{prompt_descrip}_sd_{g_idx}.png")
            g_img.save(tmp_fp)

        # --------------------------------------------------

        text_embeddings = guidance.encode_text_posneg(prompt)
        # --------------------------------------------------

        ini_num_paths = 0
        # 32, 64, 90, 100, 128
        # add_num_paths = 64 * 2

        args.num_paths = ini_num_paths + add_num_paths

        # ----------------------------------------------------------------
        pos_init_method = RandomCoordInit(
            canvas_size=[h, w], edge_margin_ratio=0.05)

        num_paths = args.num_paths

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

        # control_polygon_distance_loss_weight = ini_cpdl_weight / 2.0
        control_polygon_distance_loss_weight = 0.001

        # Run the main optimization loop
        for t in trange(args.num_iter):

            z_optimizer.zero_grad()
            affine_optimizer.zero_grad()
            color_optimizer.zero_grad()

            for _ in range(GRAD_ACC):

                # ---------------------------------------------
                # z_list[0].shape:  torch.Size([1, 24])
                z_batch = torch.stack(z_list).to(device).squeeze(1)
                # 使用模型生成点序列（批处理）
                generated_data_batch = model(
                    args_enc=None, args_dec=None, z=z_batch.unsqueeze(1).unsqueeze(2))
                generated_pts_batch = generated_data_batch["args_logits"]
                recon_data_output_batch = generated_pts_batch.squeeze(
                    1)
                # recon_data_output_batch.shape:  torch.Size([60, 32, 2])
                # ---------------------------------------------

                # -------------------------------------------
                # latent inversion
                tmp_paths_list = []
                tmp_new_points_list = []

                m_control_polygon_distance_loss = 0.0

                for _idx in range(len(z_list)):
                    # 使用z生成点序列

                    convert_points, convert_points_ini = recon_to_affine_pts(
                        recon_data_output=recon_data_output_batch[_idx], theta=theta_list[_idx], tx=tx_list[_idx], ty=ty_list[_idx], s=s_list[_idx], s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

                    # 应用仿射变换生成路径对象
                    optm_convert_path = pts_to_pathObj(convert_points)

                    tmp_paths_list.append(optm_convert_path)
                    tmp_new_points_list.append(
                        optm_convert_path.points)

                # -------------------------------------------

                m_control_polygon_distance_loss = m_control_polygon_distance_loss / \
                    num_paths * control_polygon_distance_loss_weight

                # 使用所有变换后的路径和对应颜色渲染图像
                recon_imgs, img = render_and_compose(
                    tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp="", render_func=render)

                if t % 20 == 0:
                    tmp_svg_fp = os.path.join(sv_dir, "iter_{}.svg".format(t))
                    save_paths_svg(
                        path_list=tmp_paths_list, fill_color_list=color_list, svg_path_fp=tmp_svg_fp, canvas_height=h, canvas_width=w)

                tmp_imgs = recon_imgs.repeat(NUM_AUGS, 1, 1, 1)
                im_batch = data_augs.forward(tmp_imgs)
                # -----------------------------------------------

                if t % 1000 == 0:
                    m_sdloss, viz_images = guidance.sds_vsd_grad_diffuser(text_embeddings=text_embeddings, pred_rgb=im_batch, guidance_scale=guidance_scale,  grad_scale=1.0,
                                                                          unet_phi=unet_phi, cfg_phi=1.0, generation_mode=generation_mode, cross_attention_kwargs={'scale': 1.0}, as_latent=False, save_guidance_path=True)

                    # save image
                    save_path = os.path.join(
                        vis_pred_dir, f"test_xt{t}_predx0.png")
                    save_image(viz_images, save_path)

                else:

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

                        # cur_path_m_smoothness_loss = smoothness_regularizer(cur_path_pts)
                        cur_path_m_smoothness_loss = laplacian_smoothing_loss(
                            cur_path_pts)

                        m_smoothness_loss += cur_path_m_smoothness_loss

                    m_smoothness_loss = m_smoothness_loss / \
                        len(tmp_new_points_list)
                    m_smoothness_loss = m_smoothness_loss * smoothing_loss_weight

                # ------------------------------------------------
                loss = m_sdloss

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

            if t % 10 == 0:
                print("render loss:", m_sdloss.item())
                # print("m_smoothness_loss:", m_smoothness_loss)
                print("loss:", loss.item())
                print("iteration:", t)

                # ------------------------------------------------
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

                if (h == 224):
                    area_threshold = 10
                else:
                    area_threshold = 25

                area_threshold = 5

                z_list, theta_list, s_list, tx_list, ty_list, color_list = reinitialize_paths(
                    cur_shapes=tp_shapes, cur_shape_groups=tp_shape_groups, z_list=z_list, theta_list=theta_list, tx_list=tx_list, ty_list=ty_list, s_list=s_list, color_list=color_list, pos_init_method=pos_init_method, opacity_threshold=0.05, area_threshold=area_threshold, add_new=add_new_flg)

                z_optimizer, color_optimizer, affine_optimizer, scheduler_z, scheduler_affine, scheduler_color = initialize_optimizers_and_schedulers(
                    z_list=z_list, theta_list=theta_list, s_list=s_list, tx_list=tx_list, ty_list=ty_list, color_list=color_list, args_num_iter=args.num_iter, z_lr=z_lr, color_lr=color_lr, affine_lr_initial=affine_lr_initial, theta_lr_initial=theta_lr_initial, s_lr_initial=s_lr_initial, cur_step=t, use_affine_norm=use_affine_norm)

                # 获取并打印当前的学习率
                current_lr_z = scheduler_z.get_last_lr()[0]
                current_lr_affine = scheduler_affine.get_last_lr()[0]

                # save reinit svg
                reinit_svg_path_fp = os.path.join(
                    reinit_dir, "reinit_iter_{}.svg".format(t))

                get_img_from_list(z_list=z_list, theta_list=theta_list, tx_list=tx_list, ty_list=ty_list, s_list=s_list, color_list=color_list, model=model,
                                  s_norm=s_norm, w=w, h=h, svg_path_fp=reinit_svg_path_fp, use_affine_norm=use_affine_norm, render_func=render, return_shapes=False)

            # ------------------------------------------------

            torch.cuda.empty_cache()

            ######## Do the gradient for unet_phi!!! #########
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
