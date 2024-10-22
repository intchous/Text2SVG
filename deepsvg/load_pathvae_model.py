import cv2
import PIL
import PIL.Image
import os.path as osp
import numpy.random as npr
import shutil
import cairosvg
import yaml

import os
import matplotlib.pyplot as plt
import torch
import pydiffvg
import argparse
import numpy as np


from deepsvg.my_svg_dataset_pts import Normalize, SVGDataset_nopadding, get_cubic_segments, cubics_to_pathObj, sample_bezier, sample_bezier_batch, get_cubic_segments_mask, cubic_segments_to_points

from deepsvg.json_help import sv_json

from deepsvg.model.config import _DefaultConfig
from deepsvg.model.model_pts_vae import SVGTransformer

from deepsvg.test_utils import load_model2, recon_to_affine_pts, pts_to_pathObj, save_paths_svg, render_and_compose


pydiffvg.set_print_timing(False)
# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda")
pydiffvg.set_device(device)
gamma = 1.0
render = pydiffvg.RenderFunction.apply


def load_path_vae():
    cfg = _DefaultConfig()

    yaml_fn = "test_transformer_vae_inversionz_v5_nopadding"
    yaml_fp = os.path.join("./deepsvg/config_files/", yaml_fn + ".yaml")

    # 从配置文件中加载配置数据
    with open(yaml_fp, 'r') as f:
        config_data = yaml.safe_load(f)

    # 使用配置数据更新cfg，即使cfg中没有预先定义的参数也会被加入
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

    h, w = 224, 224
    # h, w = 512, 512
    # h, w = 600, 600

    log_interval = 20
    validate_interval = 4

    log_dir = "./transformer_vae_logs/"

    signature = "ini_svgs_470510"

    absolute_base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))) + "/"

    print("absolute_base_dir: ", absolute_base_dir)

    # absolute_base_dir = "/home/zhangpeiying/research/deepsvg_relate/deepsvg/"

    svg_data_dir = os.path.join(
        absolute_base_dir + "dataset/", signature + "_cubic_single_fit/")
    svg_data_img_dir = os.path.join(
        absolute_base_dir + "dataset/", signature + "_cubic_single_img/")

    model_dir = absolute_base_dir + "transformer_vae_logs/models/"

    model = SVGTransformer(cfg)
    model = model.to(device)

    # 1-5-7: svg_emd_loss
    # 1-5-8: sample points F.mse_loss
    desc = "naive_vae_transformer_v1-5-7_" + "dataset-" + signature + "_" + "kl-" + \
        str(kl_coe) + "_" + "hd-" + str(hidden_dim) + "_" + "ld-" + str(latent_dim) + \
        "_" + "avg-" + str(cfg.avg_path_zdim) + "_" + "vae-" + \
        str(cfg.use_vae) + "_" + "sigm-" + str(cfg.use_sigmoid) + \
        "_" + "usemf-" + str(cfg.use_model_fusion) + "_" + \
        "losswl1-" + str(cfg.loss_w_l1) + "_" + "mce-" + \
        str(cfg.ModifiedConstEmbedding)

    print("desc: ", desc)

    transformer_signature = desc

    model_save_dir = os.path.join(log_dir, "models", desc)
    model_fp = os.path.join(model_save_dir, "best.pth")
    # model_fp = os.path.join(model_save_dir, "epoch_56.pth")

    # load pretrained model
    load_model2(model_fp, model)
    model.eval()

    # 是否将 affine 参数作用在归一化的points上
    use_affine_norm = False

    s_norm = Normalize(w, h)

    return model, s_norm, use_affine_norm


# --------------------------------------
def get_img_from_list(z_list, theta_list, tx_list, ty_list, s_list, color_list, model, s_norm, w=224, h=224, svg_path_fp="", use_affine_norm=False, render_func=None, return_shapes=True):

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
            recon_data_output=recon_data_output_batch[_idx], theta=theta_list[_idx]/1.0, tx=tx_list[_idx]/1.0, ty=ty_list[_idx]/1.0, s=s_list[_idx]/1.0, s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

        # 应用仿射变换生成路径对象
        optm_convert_path = pts_to_pathObj(convert_points)
        # convert_ini_path = pts_to_pathObj(convert_points_ini)

        tmp_paths_list.append(optm_convert_path)

    if (return_shapes):
        # 使用所有变换后的路径和对应颜色渲染图像
        recon_imgs, tmp_img_render, tp_shapes, tp_shape_groups = render_and_compose(
            tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp=svg_path_fp, render_func=render_func, return_shapes=return_shapes)
        return recon_imgs, tmp_img_render, tp_shapes, tp_shape_groups

    else:
        recon_imgs, tmp_img_render = render_and_compose(
            tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp=svg_path_fp, render_func=render_func, return_shapes=return_shapes)
        return recon_imgs, tmp_img_render

# --------------------------------------
