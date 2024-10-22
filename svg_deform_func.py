import os
import shutil
import numpy as np
from collections import namedtuple

import torch
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg

from utils_optm import linear_decay_lrlambda_f
from rigid_loss import procrustes_distance
from multiscale_loss import gaussian_pyramid_loss
from path_ini_util import load_target

from utils_svg import save_svg, binary_image_to_svg2
from path_ini_util import load_init_svg, path2img

from utils_match import find_max_contour_box, flood_fill, rm_mk_dir

SVG = namedtuple("SVG", "paths attributes")


def finetune_shapes(ini_shapes, ini_shape_groups, ini_point_var_fixed, img_fp, num_iter=100, gt_rsz=(224, 224), svgsign="ini", device="cuda"):

    gt = load_target(fp=img_fp, size=gt_rsz, return_rgb=False, device=device)
    h, w = gt.shape[2:]

    render = pydiffvg.RenderFunction.apply
    para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)
    lrlambda_f = linear_decay_lrlambda_f(num_iter, 0.4)
    shapes_record, shape_groups_record = [], []

    pathn = len(ini_shapes)
    cur_shapes = []
    cur_shape_groups = []
    for ki in range(pathn):
        cur_shapes.append(ini_shapes[ki])
        cur_shape_groups.append(ini_shape_groups[ki])

    cur_point_var = []
    cur_color_var = []
    for path in cur_shapes:
        path.points.requires_grad = True
        cur_point_var.append(path.points)

    for group in cur_shape_groups:
        group.fill_color.requires_grad = True
        cur_color_var.append(group.fill_color)

    # -----------------------------------------------
    shapes_record += cur_shapes
    shape_groups_record += cur_shape_groups

    para = {'point': cur_point_var, 'color': cur_color_var}
    pg = [{'params': para['point'], 'lr': 1.0},
          {'params': para['color'], 'lr': 0.01}]
    optim = torch.optim.Adam(pg)
    scheduler = LambdaLR(optim, lr_lambda=lrlambda_f, last_epoch=-1)

    procrustes_thresh = 1e-4
    for t in range(num_iter):
        optim.zero_grad()

        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            w, h, shapes_record, shape_groups_record)
        img = render(w, h, 2, 2, t, None, *scene_args)

        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + \
            para_bg * (1 - img[:, :, 3:4])

        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        # -----------------------------
        pym_loss_weight = 300.0
        pym_loss = gaussian_pyramid_loss(x, gt) * pym_loss_weight

        # -----------------------------
        if (svgsign == "ini"):
            global_procrustes_loss_weight = 0.0
        else:
            global_procrustes_loss_weight = 0.1

        m_global_procrustes_loss = 0.0
        if (global_procrustes_loss_weight > 0.0):
            for idx_path in range(len(cur_point_var)):
                ini_path_pts = ini_point_var_fixed[idx_path].to(device)
                cur_path_pts = cur_point_var[idx_path].to(device)

                cur_path_m_global_procrustes_loss = procrustes_distance(
                    ini_path_pts, cur_path_pts)

                if (cur_path_m_global_procrustes_loss > procrustes_thresh):
                    m_global_procrustes_loss += cur_path_m_global_procrustes_loss

            m_global_procrustes_loss = m_global_procrustes_loss * \
                global_procrustes_loss_weight / len(cur_point_var)

        # ------------------------------------------------
        loss = pym_loss + m_global_procrustes_loss
        loss.backward()

        optim.step()
        scheduler.step()

        for group in shape_groups_record:
            group.fill_color.data.clamp_(0.0, 1.0)

    return shapes_record, shape_groups_record, gt


def pre_contours_normd_mask(mask):
    img_mask = np.ascontiguousarray(mask, dtype=np.uint8)

    # ----------------------------
    sum_img_mask = np.sum(img_mask)
    if (sum_img_mask == 0):
        return {"sum_img_mask": 0}

    sub_corrd_arr = np.where(mask == 255)
    sub_img_m_x_mean = round(np.mean(sub_corrd_arr[0]))
    sub_img_m_y_mean = round(np.mean(sub_corrd_arr[1]))

    img_mask_contour_max_st1, img_mask_contour_max_bbox1 = find_max_contour_box(
        img_mask)
    if ((img_mask_contour_max_st1 is None)
            or (img_mask_contour_max_bbox1 is None)):
        return {"sum_img_mask": sum_img_mask, "img_mask_contour_max_st1": None, "img_mask_contour_max_bbox1": None}

    return {
        "sum_img_mask": sum_img_mask,
        # "img_mask": img_mask,
        "sub_img_m_x_mean": sub_img_m_x_mean,
        "sub_img_m_y_mean": sub_img_m_y_mean,
        "img_mask_contour_max_st1": img_mask_contour_max_st1,
        "img_mask_contour_max_bbox1": img_mask_contour_max_bbox1
    }


def mask2_path(mask, pth_color, svg_sv_fp="", experiment_dir=""):
    P = []
    A = []
    if np.sum(mask) == 0:
        return {"trans_suc": False, "cur_path_points": []}

    mask_fill_out = flood_fill(
        mask, canvas_size=(mask.shape[0], mask.shape[1]))

    # turn mask_fill_out to binary image
    mask_fill_out = mask_fill_out.astype(np.uint8)
    mask_fill_out[mask_fill_out > 0] = 255
    mask_fill_out = (mask_fill_out == 255)

    paths, attrs, svg_attrs = binary_image_to_svg2(mask_fill_out)
    for attr in attrs:
        r = pth_color[0]
        g = pth_color[1]
        b = pth_color[2]
        col = f"rgb({r},{g},{b})"
        attr["stroke"] = col
        attr["fill"] = col
    P.extend(paths)
    A.extend(attrs)
    potrace_svg = SVG(paths=P, attributes=A)
    # save the SVG
    save_svg(potrace_svg, svg_sv_fp)

    img_rec_width = mask.shape[1]
    img_rec_height = mask.shape[0]

    tmp_ca_shapes, tmp_ca_shape_groups, tmp_ca_point_var, tmp_ca_color_var = load_init_svg(
        svg_sv_fp,
        canvas_size=(img_rec_width, img_rec_height),
        trainable_stroke=False,
        requires_grad=False,
        experiment_dir=experiment_dir,
        svg_cario_dir=experiment_dir,
        add_circle=False)

    ca_cnt = 0
    for ini_ca_path in tmp_ca_shapes:
        tp_fill_color = tmp_ca_color_var[ca_cnt]
        p_img_np = path2img(ini_path=ini_ca_path,
                            fill_color_init=tp_fill_color,
                            h=img_rec_height,
                            w=img_rec_width).cpu().numpy()

        potr_path_mask = np.ascontiguousarray((p_img_np[:, :, 3] == 1.0) * 255,
                                              dtype=np.uint8)
        break

    cur_pre_contour_info = pre_contours_normd_mask(mask)
    cur_sum_img_mask = cur_pre_contour_info["sum_img_mask"]
    if (cur_sum_img_mask == 0):
        return {"trans_suc": False, "cur_path_points": []}

    cur_img_mask_contour_max_st1 = cur_pre_contour_info["img_mask_contour_max_st1"]
    cur_img_mask_contour_max_bbox1 = cur_pre_contour_info["img_mask_contour_max_bbox1"]
    cur_x, cur_y = cur_img_mask_contour_max_bbox1[0], cur_img_mask_contour_max_bbox1[1]
    cur_w, cur_h = cur_img_mask_contour_max_bbox1[2], cur_img_mask_contour_max_bbox1[3]
    # print("cur_img_mask_contour_max_bbox1: ", cur_img_mask_contour_max_bbox1)
    cur_ratio = cur_w * 1.0 / cur_h
    # print("cur_ratio = ", cur_ratio)

    if (cur_img_mask_contour_max_st1 is None or cur_img_mask_contour_max_bbox1 is None):
        return {"trans_suc": False, "cur_path_points": []}

    ref_pre_contour_info = pre_contours_normd_mask(potr_path_mask)

    ref_sum_img_mask = ref_pre_contour_info["sum_img_mask"]
    if (ref_sum_img_mask == 0):
        return {"trans_suc": False, "cur_path_points": []}

    ref_img_mask_contour_max_st1 = ref_pre_contour_info[
        "img_mask_contour_max_st1"]
    ref_img_mask_contour_max_bbox1 = ref_pre_contour_info["img_mask_contour_max_bbox1"]
    ref_x, ref_y = ref_img_mask_contour_max_bbox1[0], ref_img_mask_contour_max_bbox1[1]
    ref_w, ref_h = ref_img_mask_contour_max_bbox1[2], ref_img_mask_contour_max_bbox1[3]
    # print("ref_img_mask_contour_max_bbox1: ", ref_img_mask_contour_max_bbox1)
    ref_ratio = ref_w * 1.0 / ref_h
    # print("ref_ratio = ", ref_ratio)

    if (ref_img_mask_contour_max_st1 is None or ref_img_mask_contour_max_bbox1 is None):
        return {"trans_suc": False, "cur_path_points": []}

    ratio_diff = min(cur_ratio / ref_ratio, ref_ratio / cur_ratio)
    # print("ratio_diff: ", ratio_diff)
    if (ratio_diff < 0.7):
        return {"trans_suc": False, "cur_path_points": []}

    s_x = cur_w * 1.0 / ref_w
    s_y = cur_h * 1.0 / ref_h
    # s_reg = (s_x + s_y) / 2
    scale_matrix = torch.tensor([[s_x, 0],
                                 [0, s_y]], device="cuda", dtype=torch.float32)

    R_reg = torch.eye(2, device="cuda", dtype=torch.float32)

    t_x = cur_x - s_x * ref_x
    t_y = cur_y - s_y * ref_y
    t_reg = torch.tensor([t_x, t_y], device="cuda", dtype=torch.float32)

    cur_path_points = None
    cur_num_control_points = None
    for path in tmp_ca_shapes:
        cur_path_points = path.points
        cur_num_control_points = path.num_control_points
        break

    cur_path_points = cur_path_points.detach().cuda()
    cur_path_points_pingyi = torch.tensor(cur_path_points,
                                          dtype=torch.float32).to("cuda")

    cur_aff_left = torch.mm(scale_matrix, R_reg)
    cur_trans_matrix = torch.tensor(t_reg, device="cuda", dtype=torch.float32)
    cur_path_points = (torch.mm(cur_path_points_pingyi, cur_aff_left) +
                       cur_trans_matrix).contiguous()

    return {"trans_suc": True, "cur_path_points": cur_path_points, "cur_num_control_points": cur_num_control_points}


def process_shapes(tmp_shapes, tmp_shape_groups, experiment_dir, h=224, w=224):

    os.makedirs(experiment_dir, exist_ok=True)
    new_path_dir = os.path.join(experiment_dir, "new_path")
    rm_mk_dir(new_path_dir)

    tmp_color_var = []
    for group in tmp_shape_groups:
        tmp_color_var.append(group.fill_color)

    tmp_new_shapes = []
    tmp_new_shape_groups = []

    p_cnt = 0
    for act_idx, tmp_path in enumerate(tmp_shapes):
        tp_fill_color = tmp_color_var[act_idx]
        tar_img = path2img(ini_path=tmp_path,
                           fill_color_init=tp_fill_color, h=h, w=w)

        tmp_path_mask_alpha = (tar_img.cpu()[:, :, 3] > 0.0).numpy()
        tar_mask = np.ascontiguousarray(tmp_path_mask_alpha * 255,
                                        dtype=np.uint8)
        tar_color = [112, 128, 144]

        svg_sv_fp = os.path.join(new_path_dir, "path_" + str(p_cnt) + ".svg")
        try:
            new_points_info = mask2_path(mask=tar_mask,
                                         pth_color=tar_color,
                                         svg_sv_fp=svg_sv_fp,
                                         experiment_dir=new_path_dir)
        except Exception as e:
            continue

        if (new_points_info["trans_suc"]):
            cur_path_points = new_points_info["cur_path_points"]
            num_control_pts = new_points_info["cur_num_control_points"]

            tmp_new_path = pydiffvg.Path(
                num_control_points=torch.LongTensor(num_control_pts),
                points=cur_path_points,
                stroke_width=torch.tensor(0.0),
                is_closed=True)
            tmp_new_shapes.append(tmp_new_path)

            fill_color_init = tmp_color_var[act_idx]
            tm_path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([p_cnt]),
                fill_color=fill_color_init,
                stroke_color=fill_color_init,
                # use_even_odd_rule=False
            )
            tmp_new_shape_groups.append(tm_path_group)

        else:
            tmp_new_shapes.append(tmp_shapes[act_idx])
            tmp_new_shape_groups.append(tmp_shape_groups[act_idx])

        p_cnt += 1

    del_dir = experiment_dir
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)

    return tmp_new_shapes, tmp_new_shape_groups, tmp_color_var
