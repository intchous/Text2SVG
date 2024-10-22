import os
import argparse
import random
import shutil
import numpy as np
import PIL
import yaml

from skimage import morphology
import pydiffvg

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import json_help
from json_help import dict_to_nonedict

from path_ini_util import load_init_svg, path2img, save_path_svg
from utils_svg import save_svg, to_svg
# from utils_match import rm_mk_dir, judge_mask, do_xy_transform
# from utils_preseg import get_segmentation, toposort_path
from svg_deform_func import finetune_shapes, process_shapes


# from optim_utils.util_svg import (
#     get_cubic_segments_from_points,
#     sample_bezier,
#     make_clockwise,
#     sample_points_by_length_distribution
# )

from optim_utils.util_pregroup import get_group, toposort_path, rm_mk_dir, judge_mask, do_xy_transform


from losses import laplacian_smoothing_loss
from utils_optm import linear_decay_lrlambda_f

from deepsvg.my_svg_dataset_pts import Normalize, SVGDataset_nopadding, get_cubic_segments, sample_bezier

from deepsvg.model.loss import svg_emd_loss, get_target_distr, make_clockwise
from deepsvg.model.config import _DefaultConfig
from deepsvg.model.model_pts_vae import SVGTransformer
from deepsvg.test_utils import load_model2, recon_to_affine_pts, pts_to_pathObj, save_paths_svg, render_and_compose


import warnings
warnings.filterwarnings("ignore")


pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(torch.cuda.is_available())
device = torch.device("cuda")
pydiffvg.set_device(device)
render = pydiffvg.RenderFunction.apply
gamma = 1.0


def convert_path_cubic(cur_path):
    # 第一个点是M
    idx_pts = 0
    pre_points = cur_path.points[0]
    new_points = [[pre_points[0], pre_points[1]]]
    new_num_control_points = []

    for num_i in cur_path.num_control_points:
        if (num_i == 0):
            idx_pts += 1
            idx_pts = idx_pts % len(cur_path.points)

            new_points.extend([[pre_points[0], pre_points[1]], [cur_path.points[idx_pts][0], cur_path.points[idx_pts][1]], [
                              cur_path.points[idx_pts][0], cur_path.points[idx_pts][1]]])

        else:
            idx_pts += 3
            idx_pts = idx_pts % len(cur_path.points)

            new_points.extend([[cur_path.points[idx_pts-2][0], cur_path.points[idx_pts-2][1]], [cur_path.points[idx_pts-1]
                              [0], cur_path.points[idx_pts-1][1]], [cur_path.points[idx_pts][0], cur_path.points[idx_pts][1]]])

        pre_points = cur_path.points[idx_pts]
        new_num_control_points.append(2)

    # list to tensor
    new_points = torch.tensor(new_points, dtype=torch.float32)
    new_num_control_points = torch.LongTensor(new_num_control_points)

    tmp_path = pydiffvg.Path(
        num_control_points=new_num_control_points,
        points=new_points,
        stroke_width=cur_path.stroke_width,
        is_closed=cur_path.is_closed)

    num_pts = tmp_path.points.shape[0]
    assert ((num_pts-1) % 3 == 0)

    # TODO: 即使当前tmp_path的num_pts是3k+1, 后面diffvg保存成svg时, 会自动优化掉最后一个点, 变成3k
    return num_pts, tmp_path


def get_z_from_circle(absolute_base_dir, cfg, model=None):
    rd_fp_list = ["2719851_cubic_3_r0.svg"]

    # dataset_h = 224
    # dataset_w = 224
    dataset_h = h
    dataset_w = w
    signature = "ini_svgs_470510"
    svg_data_img_dir = os.path.join(
        absolute_base_dir, "vae_dataset", signature + "_cubic_single_img/")

    tmp_svg_dataset_test = SVGDataset_nopadding(
        directory=absolute_base_dir + "vae_dataset/ini_svgs_470510_cubic_single_fit/", h=dataset_h, w=dataset_w, fixed_length=cfg.max_pts_len_thresh, file_list=rd_fp_list, img_dir=svg_data_img_dir, transform=Normalize(dataset_w, dataset_h), use_model_fusion=cfg.use_model_fusion)

    tmp_test_loader = DataLoader(
        tmp_svg_dataset_test, batch_size=1, shuffle=False)

    # 获得svg的latent vector
    # with torch.no_grad():
    for i, batch_data in enumerate(tmp_test_loader):
        points_batch = batch_data["points"]
        filepaths = batch_data["filepaths"]
        path_imgs = batch_data["path_img"]
        path_imgs = path_imgs.to(device)

        bat_s, _, _, _ = batch_data["cubics"].shape
        cubics_batch_fl = batch_data["cubics"].view(bat_s, -1, 2)
        data_pts = cubics_batch_fl.to(device)

        data_input = data_pts.unsqueeze(1)
        # data_input.shape:  torch.Size([1, 1, 62, 2])

        output = model(args_enc=data_input,
                       args_dec=data_input, ref_img=path_imgs)

        z1 = model.latent_z.detach().cpu().clone()

    return z1.squeeze(0).squeeze(1)


def initialize_optimizers_and_schedulers(z_list, theta_list, s_list, tx_list, ty_list, color_list, args_num_iter, z_lr=None, color_lr=None, affine_lr=None, theta_lr=None, s_lr=None, decay_fac=0.6):

    z_optimizer = optim.Adam(z_list, lr=z_lr)
    color_optimizer = optim.Adam(color_list, lr=color_lr)

    affine_optimizer = torch.optim.Adam([
        # 0.1, 0.01, 0.05
        {'params': theta_list, 'lr': theta_lr},
        {'params': s_list, 'lr': s_lr},
        {'params': tx_list, 'lr': affine_lr},
        {'params': ty_list, 'lr': affine_lr}
    ])

    lrlambda_f_z = linear_decay_lrlambda_f(args_num_iter, decay_fac)
    lrlambda_f_affine = linear_decay_lrlambda_f(args_num_iter, decay_fac)
    lrlambda_f_color = linear_decay_lrlambda_f(args_num_iter, decay_fac)

    scheduler_z = LambdaLR(z_optimizer, lr_lambda=lrlambda_f_z, last_epoch=-1)
    scheduler_affine = LambdaLR(
        affine_optimizer, lr_lambda=lrlambda_f_affine, last_epoch=-1)
    scheduler_color = LambdaLR(
        color_optimizer, lr_lambda=lrlambda_f_color, last_epoch=-1)

    return z_optimizer, color_optimizer, affine_optimizer, scheduler_z, scheduler_affine, scheduler_color


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

        optm_convert_path = pts_to_pathObj(convert_points)
        tmp_paths_list.append(optm_convert_path)

    if (return_shapes):
        # 使用所有变换后的路径和对应颜色渲染图像
        recon_img, tmp_img_render, tp_shapes, tp_shape_groups = render_and_compose(
            tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp=svg_path_fp, render_func=render, return_shapes=return_shapes)
        return recon_img, tmp_img_render, tp_shapes, tp_shape_groups

    else:
        recon_img, tmp_img_render = render_and_compose(
            tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, svg_path_fp=svg_path_fp, render_func=render_func, return_shapes=return_shapes)
        return recon_img, tmp_img_render


def get_cubic_segments_from_points(points):
    cubics = []
    idx = 0
    total_points = points.shape[0]
    seg_num = points.shape[0] // 3

    for sg_ in range(seg_num):
        pt1 = points[idx]
        pt2 = points[idx + 1]
        pt3 = points[idx + 2]
        pt4 = points[(idx + 3) % total_points]

        cubics.append(pt1)
        cubics.append(pt2)
        cubics.append(pt3)
        cubics.append(pt4)

        idx += 3

    # total_points/3*4
    cubics = torch.stack(cubics).view(-1, 4, 2)
    return cubics


def kl_divergence(src_z):
    src_mean = torch.mean(src_z, dim=-1)
    src_std = torch.std(src_z, dim=-1)
    kl_div = 0.5 * torch.sum(src_std**2 + src_mean **
                             2 - 1 - torch.log(src_std**2), dim=-1)

    return kl_div.mean()


def latent_inversion(ctrl_pts_cubic_list, cfg, model, s_norm, num_paths=10, device="cuda", num_epochs=200, z_lr=None, affine_lr=None, theta_lr=None, s_lr=None):

    print("latent inversion...")
    model.eval()

    each_cubic_sample_num = 8
    tar_sampled_points_clockwise_list = []
    p_target_sub_list = []
    matching_list = []

    for ctrl_pts_cubic in ctrl_pts_cubic_list:

        tar_cubics = ctrl_pts_cubic

        tar_sampled_points = sample_bezier(
            tar_cubics, each_cubic_sample_num).to(device)

        tar_sampled_points_clockwise = make_clockwise(tar_sampled_points)
        tar_sampled_points_clockwise_list.append(tar_sampled_points_clockwise)

        pred_sampled_points_n = cfg.max_total_len // 4 * each_cubic_sample_num
        # pred_sampled_points_n: 60

        p_target_sub, matching = get_target_distr(
            p_target=tar_sampled_points_clockwise, n=pred_sampled_points_n, device=tar_sampled_points_clockwise.device)

        p_target_sub_list.append(p_target_sub)
        matching_list.append(matching)

    # 初始化z
    z_list = [get_z_from_circle(absolute_base_dir="./", cfg=cfg, model=model).to(
        device).requires_grad_(True) for _ in range(num_paths)]

    theta_list = [torch.tensor(0.0).to(device).requires_grad_(True)
                  for _ in range(num_paths)]

    # 0.2, 0.25, 0.3
    s_list = [torch.tensor(0.2).to(device).requires_grad_(True)
              for _ in range(num_paths)]

    tx_list = [torch.tensor(0.0).to(device).requires_grad_(True)
               for _ in range(num_paths)]
    ty_list = [torch.tensor(0.0).to(device).requires_grad_(True)
               for _ in range(num_paths)]

    z_optimizer, color_optimizer, affine_optimizer, scheduler_z, scheduler_affine, scheduler_color = initialize_optimizers_and_schedulers(
        z_list=z_list, theta_list=theta_list, s_list=s_list, tx_list=tx_list, ty_list=ty_list, color_list=[torch.tensor(0.0).to(device)], args_num_iter=num_epochs, z_lr=z_lr, color_lr=0, affine_lr=affine_lr, theta_lr=theta_lr, s_lr=s_lr)

    best_z_list = None
    best_s_list = None
    best_theta_list = None
    best_tx_list = None
    best_ty_list = None

    best_loss = float('inf')
    m_smoothness_loss_weight = 3.0
    m_kl_loss_weight = 0.1

    for epoch in range(num_epochs):
        # z_list[0].shape:  torch.Size([1, 24])
        z_batch = torch.stack(z_list).to(device).squeeze(1)
        generated_data_batch = model(
            args_enc=None, args_dec=None, z=z_batch.unsqueeze(1).unsqueeze(2))
        generated_pts_batch = generated_data_batch["args_logits"]
        recon_data_output_batch = generated_pts_batch.squeeze(1)
        # recon_data_output_batch.shape:  torch.Size([60, 32, 2])
        # ---------------------------------------------

        m_kl_loss = 0.0
        if (m_kl_loss_weight > 0):
            m_kl_loss = kl_divergence(z_batch) * m_kl_loss_weight

        tmp_paths_list = []
        total_svg_emd_loss = 0.0
        m_smoothness_loss = 0.0
        for idx in range(num_paths):
            recon_data_output = recon_data_output_batch[idx]
            # recon_data_output.shape:  torch.Size([40, 2])

            convert_points, convert_points_ini = recon_to_affine_pts(
                recon_data_output=recon_data_output, theta=theta_list[idx], tx=tx_list[idx], ty=ty_list[idx], s=s_list[idx], s_norm=s_norm, h=h, w=w, use_affine_norm=False)

            # -----------------------------------------------
            if (m_smoothness_loss_weight > 0):

                cur_path_m_smoothness_loss = laplacian_smoothing_loss(
                    convert_points)

                n_points = convert_points.size(0)
                average_smoothness_loss = cur_path_m_smoothness_loss / n_points

                m_smoothness_loss = m_smoothness_loss + average_smoothness_loss
            # -----------------------------------------------

            optm_convert_path = pts_to_pathObj(convert_points)
            tmp_paths_list.append(optm_convert_path)

            convert_cubics = get_cubic_segments_from_points(convert_points)
            ini_cubics = convert_cubics.view(-1, 4, 2)

            pred_sampled_points = sample_bezier(
                ini_cubics, each_cubic_sample_num)
            # pred_sampled_points.shape:  torch.Size([60, 2])

            m_svg_emd_loss = svg_emd_loss(
                p_pred=pred_sampled_points,
                p_target=tar_sampled_points_clockwise_list[idx],
                p_target_sub=p_target_sub_list[idx],
                matching=matching_list[idx],
            )

            total_svg_emd_loss = total_svg_emd_loss + m_svg_emd_loss

        # -----------------------------------------------
        m_smoothness_loss = m_smoothness_loss * m_smoothness_loss_weight / num_paths

        total_svg_emd_loss = total_svg_emd_loss / num_paths
        loss = total_svg_emd_loss + m_smoothness_loss + m_kl_loss

        # 如果当前损失小于最佳损失，则更新best_z和best_img
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z_list = z_list
            best_s_list = s_list
            best_theta_list = theta_list
            best_tx_list = tx_list
            best_ty_list = ty_list

        z_optimizer.zero_grad()
        affine_optimizer.zero_grad()

        loss.backward()

        z_optimizer.step()
        affine_optimizer.step()

        scheduler_z.step()
        scheduler_affine.step()

    return best_z_list, best_s_list, best_theta_list, best_tx_list, best_ty_list, best_loss


# -------------------------------------------------------------
def img_optm(target_img, z_list, color_list, s_list, theta_list, tx_list, ty_list, model, s_norm, num_paths=10, device="cuda", num_epochs=200, z_lr=None, color_lr=None, affine_lr=None, theta_lr=None, s_lr=None, optm_color_img=True):

    print("img optm...")
    model.eval()
    use_affine_norm = False

    z_optimizer, color_optimizer, affine_optimizer, scheduler_z, scheduler_affine, scheduler_color = initialize_optimizers_and_schedulers(
        z_list=z_list, theta_list=theta_list, s_list=s_list, tx_list=tx_list, ty_list=ty_list, color_list=color_list, args_num_iter=num_epochs, z_lr=z_lr, color_lr=color_lr, affine_lr=affine_lr, theta_lr=theta_lr, s_lr=s_lr)

    best_z_list = None
    best_s_list = None
    best_theta_list = None
    best_tx_list = None
    best_ty_list = None
    best_color_list = None
    best_img = None
    best_loss = float('inf')

    pym_loss_weight = 300.0
    mse_loss_weight = 1.0
    # 1.0
    m_smoothness_loss_weight = 3.0

    for epoch in range(num_epochs):
        z_batch = torch.stack(z_list).to(device).squeeze(1)
        generated_data_batch = model(
            args_enc=None, args_dec=None, z=z_batch.unsqueeze(1).unsqueeze(2))
        generated_pts_batch = generated_data_batch["args_logits"]
        recon_data_output_batch = generated_pts_batch.squeeze(1)
        # recon_data_output_batch.shape:  torch.Size([60, 32, 2])
        # ---------------------------------------------

        tmp_paths_list = []
        tmp_new_points_list = []
        m_smoothness_loss = 0.0
        m_pym_loss = 0.0

        for idx in range(num_paths):
            recon_data_output = recon_data_output_batch[idx]
            # recon_data_output.shape:  torch.Size([40, 2])

            convert_points, convert_points_ini = recon_to_affine_pts(
                recon_data_output=recon_data_output, theta=theta_list[idx], tx=tx_list[idx], ty=ty_list[idx], s=s_list[idx], s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)
            # convert_points.shape:  torch.Size([30, 2]) (0-224)
            # -----------------------------------------------

            if (m_smoothness_loss_weight > 0):
                cur_path_m_smoothness_loss = laplacian_smoothing_loss(
                    convert_points)

                n_points = convert_points.size(0)
                average_smoothness_loss = cur_path_m_smoothness_loss / n_points

                m_smoothness_loss = m_smoothness_loss + average_smoothness_loss
            # -----------------------------------------------
            optm_convert_path = pts_to_pathObj(convert_points)

            tmp_paths_list.append(optm_convert_path)
            tmp_new_points_list.append(optm_convert_path.points)

        # -----------------------------------------------
        m_smoothness_loss = m_smoothness_loss * m_smoothness_loss_weight / num_paths

        recon_img, tmp_img_render = render_and_compose(
            tmp_paths_list=tmp_paths_list, color_list=color_list, w=w, h=h, render_func=render)

        if (recon_img.requires_grad == False):
            continue

        # -----------------------------------------------
        mse_loss = torch.nn.functional.mse_loss(
            recon_img, target_img) * mse_loss_weight

        loss = mse_loss

        if (m_smoothness_loss_weight > 0):
            loss = loss + m_smoothness_loss
        # -----------------------------------------------

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_z_list = z_list
            best_s_list = s_list
            best_theta_list = theta_list
            best_tx_list = tx_list
            best_ty_list = ty_list
            best_color_list = color_list
            best_img = tmp_img_render.detach().clone()

            best_tp_shapes, best_tp_shape_groups = save_paths_svg(
                path_list=tmp_paths_list, fill_color_list=color_list, canvas_height=h, canvas_width=w)
            # --------------------------------------

        color_optimizer.zero_grad()
        affine_optimizer.zero_grad()
        z_optimizer.zero_grad()

        loss.backward()

        z_optimizer.step()
        affine_optimizer.step()
        if (optm_color_img):
            color_optimizer.step()

        scheduler_z.step()
        scheduler_affine.step()
        if (optm_color_img):
            scheduler_color.step()

    return best_tp_shapes, best_tp_shape_groups, best_z_list, best_s_list, best_theta_list, best_tx_list, best_ty_list, best_color_list, best_img, best_loss


def imgs_dir_to_svg(signature, pregroup_opt, test_save_dir, tar_img_dir, gt_rsz=(224, 224), svgsign="ini"):

    tar_img_list = os.listdir(tar_img_dir)
    random.shuffle(tar_img_list)

    # ------------------------------------------
    tar_img_seg_dir = os.path.join(
        test_save_dir, "tar_" + signature + "_img_seg/")
    rm_mk_dir(tar_img_seg_dir)

    experiment_dir = os.path.join(test_save_dir, "tar_" + signature + "_expm/")
    # rm_mk_dir(experiment_dir)

    tpsvg_dir = os.path.join(test_save_dir, "tar_" + signature + "_out_proc/")
    os.makedirs(tpsvg_dir, exist_ok=True)

    tar_img_mask_dir = os.path.join(
        test_save_dir, "tar_" + signature + "_img_mask/")
    rm_mk_dir(tar_img_mask_dir)

    tar_mask_potrace_path_dir = os.path.join(
        test_save_dir, "tar_" + signature + "_mask_potrace_path/")
    rm_mk_dir(tar_mask_potrace_path_dir)

    # --------------------------------------------
    fg_overlap_thresh = 0.8

    rag_sigma = pregroup_opt["rag_sigma"]
    rag_mode = pregroup_opt["rag_mode"]
    rag_connectivity = pregroup_opt["rag_connectivity"]
    seg_sigma = pregroup_opt["seg_sigma"]
    seg_max_dist = pregroup_opt["seg_max_dist"]
    seg_ratio = pregroup_opt["seg_ratio"]
    seg_kernel_size = pregroup_opt["seg_kernel_size"]
    mer_thresh = pregroup_opt["mer_thresh"]
    is_merge = pregroup_opt["is_merge"]

    canvas_height = gt_rsz[0]
    canvas_width = gt_rsz[1]
    # --------------------------------------------

    for fn in tar_img_list:
        infile = os.path.join(tar_img_dir, fn)

        if os.path.isdir(infile):
            continue

        im_pre, im_ext = os.path.splitext(fn)
        if (im_ext != ".png" and im_ext != ".jpg" and im_ext != ".jpeg"):
            continue

        # --------------------------------------------
        # TODO: test
        tar_ftsvg_fp = os.path.join(
            tpsvg_dir, im_pre + "_" + svgsign + '_optm.svg')
        if (os.path.exists(tar_ftsvg_fp)):
            continue
        # --------------------------------------------
        print("im_fn = ", fn)

        num_iter = 50

        img_pil = PIL.Image.open(infile).convert('RGBA').resize(
            gt_rsz, PIL.Image.Resampling.BICUBIC)

        # --------------------------------------------
        # img_fg_mask = remove(img_pil, only_mask=True)
        # --------------------------------------------

        img_fp = os.path.join(tar_img_dir, fn)

        img, seg = get_group(
            img_fp=img_fp,
            gt_rsz=gt_rsz,
            seg_max_dist=seg_max_dist,
            seg_ratio=seg_ratio,
            seg_kernel_size=seg_kernel_size,
            seg_sigma=seg_sigma,
            rag_sigma=rag_sigma,
            rag_connectivity=rag_connectivity,
            rag_mode=rag_mode,
            mer_thresh=mer_thresh,
            is_merge=is_merge)

        # --------------------------------
        svg = to_svg(img, seg)
        ini_svg_fp = os.path.join(experiment_dir, im_pre + ".svg")
        save_svg(svg, ini_svg_fp)
        # --------------------------------

        # --------------------------------
        img_rec_width = img.shape[1] + int(img.shape[1] / 10) * 2
        img_rec_height = img.shape[0] + int(img.shape[0] / 10) * 2

        tmp_ca_shapes, tmp_ca_shape_groups, tmp_ca_point_var, tmp_ca_color_var = load_init_svg(
            ini_svg_fp,
            canvas_size=(img_rec_width, img_rec_height),
            trainable_stroke=False,
            requires_grad=False,
            experiment_dir=experiment_dir,
            scale_fac=1.334,
            svg_cario_dir=experiment_dir,
            add_circle=False)

        selem_example = morphology.square(pregroup_opt["morph_kernel_size"])
        # --------------------------------------------
        mask_dir = os.path.join(
            test_save_dir, "tar_" + signature + "_mask", im_pre)
        rm_mk_dir(mask_dir)

        potrace_path_dir = os.path.join(tar_mask_potrace_path_dir, im_pre)
        rm_mk_dir(potrace_path_dir)

        ca_cnt = 0
        cur_tar_img_svg_path_info = []
        cur_tar_shapes_ini = []
        cur_tar_shapes_ini_groups = []
        act_cnt = 0
        for ini_ca_path in tmp_ca_shapes:
            cur_path_points = ini_ca_path.points.detach().cuda()
            if (cur_path_points.shape[0] == 0):
                ca_cnt += 1
                continue

            # --------------------------------------
            left = int(img.shape[1] / 10)
            right = img.shape[1] + int(img.shape[1] / 10) - 1
            top = int(img.shape[0] / 10)
            bottom = img.shape[0] + int(img.shape[0] / 10) - 1

            diff_x = -top
            diff_y = -left

            cur_path_points_pingyi = do_xy_transform(
                pts_set_src=cur_path_points, dx=diff_y, dy=diff_x)
            cur_path_points_pingyi = torch.tensor(
                cur_path_points_pingyi, dtype=torch.float32).to("cuda")

            ini_ca_path.points = cur_path_points_pingyi
            tp_fill_color = tmp_ca_color_var[ca_cnt]

            # --------------------------------------
            p_img_np = path2img(ini_path=ini_ca_path,
                                fill_color_init=tp_fill_color,
                                h=canvas_height,
                                w=canvas_width).cpu().numpy()

            # 半透明的像素也算
            tmp_path_mask_alpha = (p_img_np[:, :, 3] > 0)
            # ------------------------------------------------------

            area_th = pregroup_opt["area_th"]
            ratio_th = pregroup_opt["ratio_th"]

            if (judge_mask(mask=tmp_path_mask_alpha, gt_rsz=gt_rsz, area_th=area_th, ratio_th=ratio_th, keep_backgrd=False, img_np=p_img_np) == False):

                ca_cnt += 1
                # os.remove(tp_svg_path_fp)
                continue
            # ------------------------------------------------------

            # -----------------------------------------------
            # 判断是否属于前景
            fg_overlap = tmp_path_mask_alpha
            if (np.sum(fg_overlap) / np.sum(tmp_path_mask_alpha) < fg_overlap_thresh):
                ca_cnt += 1
                # os.remove(tp_svg_path_fp)
                continue
            # -----------------------------------------------

            # -----------------------------------------------
            opened_seg = morphology.opening(
                tmp_path_mask_alpha, selem_example)
            morph_opened_sum = np.sum(opened_seg)

            if (morph_opened_sum == 0):
                ca_cnt += 1
                continue
            # -----------------------------------------------

            tp_svg_path_fp = os.path.join(
                potrace_path_dir, im_pre + "_seg_" + str(ca_cnt) + ".svg")
            save_path_svg(ini_ca_path,
                          svg_path_fp=tp_svg_path_fp,
                          fill_color=tp_fill_color,
                          canvas_height=canvas_height,
                          canvas_width=canvas_width)

            p_img_np_pil = PIL.Image.fromarray(
                (p_img_np * 255).astype(np.uint8), "RGBA")
            tp_svg_path_png_fp = os.path.join(
                potrace_path_dir, im_pre + "_seg_" + str(ca_cnt) + ".png")
            p_img_np_pil.save(tp_svg_path_png_fp)

            # -----------------------------------------------
            cur_tar_img_svg_path_info.append({
                "tar_svg_path_mask_sub_fp": tp_svg_path_png_fp,
                "tar_svg_path_sub_fp": tp_svg_path_fp,
                "fill_color_target": tp_fill_color,
            })

            cur_tar_shapes_ini.append(ini_ca_path)
            cur_path_group = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor(
                [act_cnt]),
                fill_color=tp_fill_color,
                use_even_odd_rule=False)
            cur_tar_shapes_ini_groups.append(cur_path_group)

            ca_cnt += 1
            act_cnt += 1

        assert ca_cnt == len(tmp_ca_shapes)

        toposort_cur_tar_img_svg_path_info, cur_tar_shapes_ini, cur_tar_shapes_ini_groups = toposort_path(
            cur_tar_img_svg_path_info)

        ini_point_var_fixed = []
        for path in cur_tar_shapes_ini:
            ini_point_var_fixed.append(path.points.clone().detach())

        tar_shapes_ft, tar_shapes_ft_groups, gt_img_tensor = finetune_shapes(
            ini_shapes=cur_tar_shapes_ini, ini_shape_groups=cur_tar_shapes_ini_groups, ini_point_var_fixed=ini_point_var_fixed, img_fp=infile, num_iter=num_iter, gt_rsz=gt_rsz, svgsign=svgsign)
        # gt_img_tensor.shape torch.Size([1, 3, 224, 224])

        post_dir = os.path.join(test_save_dir, svgsign + "_post")
        tar_shapes_proc, tar_shapes_proc_groups, tar_shapes_color_var = process_shapes(
            tmp_shapes=tar_shapes_ft, tmp_shape_groups=tar_shapes_ft_groups, experiment_dir=post_dir, h=gt_rsz[0], w=gt_rsz[1])

        # -----------------------------------------------
        num_paths = len(tar_shapes_proc)
        print("num_paths: ", num_paths)

        if (num_paths > 512):
            continue

        # tp_shapes = tar_shapes_proc
        # tp_shapes_groups = tar_shapes_proc_groups

        tp_shapes = cur_tar_shapes_ini
        tp_shapes_groups = cur_tar_shapes_ini_groups

        ctrl_pts_cubic_list = []
        for _path in tp_shapes:
            num_pts, tmp_path = convert_path_cubic(_path)
            tmp_cubic_segments = get_cubic_segments(tmp_path)
            ctrl_pts_cubic_list.append(tmp_cubic_segments)

        best_z_list, best_s_list, best_theta_list, best_tx_list, best_ty_list, _ = latent_inversion(
            ctrl_pts_cubic_list=ctrl_pts_cubic_list, cfg=cfg, model=model, s_norm=s_norm, num_paths=num_paths, device=device, num_epochs=260, z_lr=seq_z_lr,  affine_lr=seq_affine_lr, theta_lr=seq_theta_lr, s_lr=seq_s_lr)

        if (best_z_list is None):
            continue

        # ---------------------------------------------
        target_img = gt_img_tensor
        if (optm_color_img):
            for t_color in tar_shapes_color_var:
                t_color.requires_grad = True

        tp_shapes, tp_shapes_groups, best_z_list, best_s_list, best_theta_list, best_tx_list, best_ty_list, _, _, _ = img_optm(
            target_img=target_img, z_list=best_z_list, color_list=tar_shapes_color_var, s_list=best_s_list, theta_list=best_theta_list, tx_list=best_tx_list, ty_list=best_ty_list,  model=model, s_norm=s_norm, num_paths=num_paths, device=device, num_epochs=80, z_lr=img_z_lr, color_lr=color_lr, affine_lr=img_affine_lr, theta_lr=img_theta_lr, s_lr=img_s_lr, optm_color_img=optm_color_img)

        tar_ftsvg_fp = os.path.join(
            tpsvg_dir, im_pre + "_" + svgsign + '_optm.svg')
        pydiffvg.save_svg(tar_ftsvg_fp, canvas_width,
                          canvas_height, tp_shapes, tp_shapes_groups)

        # -----------------------------------------------


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python img_to_svg.py --signature svg_png --svgsign ini --sz 224

    # CUDA_VISIBLE_DEVICES=2 python img_to_svg.py --signature svg_png --svgsign tpsort --sz 224

    parser = argparse.ArgumentParser()
    parser.add_argument("--signature", type=str, default="svg_png")
    parser.add_argument("--svgsign", type=str,
                        default="ini", help="ini or tpsort")
    parser.add_argument("--sz", type=int, default=224)
    args = parser.parse_args()

    h = args.sz
    w = args.sz
    seg_size = (h, w)

    img_match_param_fp = "./img_group_param_" + str(h) + ".yaml"
    pregroup_opt = json_help.parse(img_match_param_fp)
    pregroup_opt = dict_to_nonedict(pregroup_opt)

    test_save_dir = "./test_dataset"
    tar_img_dir = os.path.join(test_save_dir, args.signature)

    save_cubic_svg_dir = os.path.join(
        test_save_dir, "tar_" + args.signature + "_out_proc/")
    os.makedirs(save_cubic_svg_dir, exist_ok=True)

    # -------------------------------------------------
    cfg = _DefaultConfig()
    yaml_fp = "./path_vae_optm.yaml"

    with open(yaml_fp, 'r') as f:
        config_data = yaml.safe_load(f)

    # 使用配置数据更新cfg，即使cfg中没有预先定义的参数也会被加入
    for key, value in config_data.items():
        setattr(cfg, key, value)

    # ---------------------------------------
    cfg.img_latent_dim = int(cfg.d_img_model / 64.0)
    cfg.vq_edim = int(cfg.dim_z / cfg.vq_comb_num)

    input_dim = cfg.n_args
    output_dim = cfg.n_args
    hidden_dim = cfg.d_model
    latent_dim = cfg.dim_z
    max_pts_len_thresh = cfg.max_pts_len_thresh
    kl_coe = cfg.kl_coe

    batch_size = cfg.batch_size
    num_epochs = 200
    learning_rate = 0.001

    log_interval = 20
    validate_interval = 20

    log_dir = "./transformer_vae_logs/"
    signature = "ini_svgs_470510"

    # 获取当前的 absolute_base_dir
    # absolute_base_dir = os.path.dirname(os.path.dirname(
    #     os.path.dirname(os.path.abspath(__file__)))) + "/"
    # print("absolute_base_dir: ", absolute_base_dir)
    absolute_base_dir = "./"

    svg_data_dir = os.path.join(
        absolute_base_dir, "vae_dataset", signature + "_cubic_single_fit/")
    svg_data_img_dir = os.path.join(
        absolute_base_dir, "vae_dataset", signature + "_cubic_single_img/")

    color_black = torch.FloatTensor([0, 0, 0, 1]).to("cuda")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SVGTransformer(cfg)
    model = model.to(device)

    desc = "naive_vae_transformer_v1-5-7_" + "dataset-" + signature + "_" + "kl-" + str(kl_coe) + "_" + "hd-" + str(hidden_dim) + "_" + "ld-" + str(latent_dim) + "_" + "avg-" + str(cfg.avg_path_zdim) + "_" + "vae-" + \
        str(cfg.use_vae) + "_" + "sigm-" + str(cfg.use_sigmoid) + \
        "_" + "usemf-" + str(cfg.use_model_fusion) + "_" + \
        "losswl1-" + str(cfg.loss_w_l1) + "_" + "mce-" + \
        str(cfg.ModifiedConstEmbedding)

    print("desc: ", desc)

    transformer_signature = desc
    model_save_dir = os.path.join("vae_model", desc)
    model_fp = os.path.join(model_save_dir, "best.pth")

    # load pretrained model
    load_model2(model_fp, model)
    model.eval()

    s_norm = Normalize(w, h)
    # -------------------------------------------------

    color_lr = 0.01
    # 是否优化颜色: 彩色图片True, 黑白图片False
    optm_color_img = True

    lr_fac = 10.0
    seq_z_lr = 0.1
    seq_affine_lr = 0.5 * lr_fac
    seq_theta_lr = 0.006 * lr_fac
    seq_s_lr = 0.05 * lr_fac

    lr_fac = 0.1
    img_z_lr = 0.08
    img_affine_lr = 0.5 * lr_fac
    img_theta_lr = 0.006 * lr_fac
    img_s_lr = 0.007 * lr_fac

    # ----------------------------------------------------
    imgs_dir_to_svg(signature=args.signature,
                    pregroup_opt=pregroup_opt, test_save_dir=test_save_dir, tar_img_dir=tar_img_dir, gt_rsz=seg_size, svgsign=args.svgsign)

    del_dir = os.path.join(test_save_dir, "tar_" +
                           args.signature + "_img_mask/")
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
    del_dir = os.path.join(test_save_dir, "tar_" + args.signature + "_mask/")
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
    del_dir = os.path.join(test_save_dir, "tar_" +
                           args.signature + "_img_seg/")
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
    del_dir = os.path.join(test_save_dir, "tar_" +
                           args.signature + "_mask_potrace_path/")
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
    del_dir = os.path.join(test_save_dir, "tar_" +
                           args.signature + "_img_mask_ovlp/")
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
