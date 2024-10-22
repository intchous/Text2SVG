import os
import numpy as np
import argparse
import shutil
import PIL
from skimage import segmentation
from skimage.future import graph
from skimage.segmentation import mark_boundaries
import cv2
import graphlib
import pydiffvg

import torch

from util_svg import _merge_mean_color, _weight_mean_color


def rm_mk_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def judge_mask(mask,
               gt_rsz=(224, 224),
               area_th=5,
               ratio_th=100.0,
               keep_backgrd=False,
               img_np=None,
               margin=2):
    # mask: 0,1
    if np.all(mask == 0):
        return False

    if (np.sum(mask == True) < area_th):
        return False

    msk_x_sum = np.sum(mask, axis=0)
    msk_y_sum = np.sum(mask, axis=1)
    msk_x_max = np.max(msk_x_sum)
    msk_y_max = np.max(msk_y_sum)

    if (max(msk_x_max, msk_y_max) / min(msk_x_max, msk_y_max) > ratio_th):
        return False

    if (keep_backgrd == False):

        if img_np is not None:
            c_lu = img_np[margin, margin]
            c_ru = img_np[margin, gt_rsz[1] - 1 - margin]
            c_lb = img_np[gt_rsz[0] - 1 - margin, margin]
            c_rb = img_np[gt_rsz[0] - 1 - margin, gt_rsz[1] - 1 - margin]

            corners = [c_lu, c_ru, c_lb, c_rb]
            wh_threshold = 0.96

            if any(np.all(np.array(corner[:3]) > wh_threshold) for corner in corners):
                return False

        else:
            c_lu = mask[margin][margin]
            c_ru = mask[margin][gt_rsz[1] - 1 - margin]
            c_lb = mask[gt_rsz[0] - 1 - margin][margin]
            c_rb = mask[gt_rsz[0] - 1 - margin][gt_rsz[1] - 1 - margin]

            if (c_lu or c_ru or c_lb or c_rb):
                return False

    return True


def do_affine_transform(pts_set_src, s_reg, R_reg, t_reg):
    cur_aff_left = torch.tensor(s_reg * R_reg,
                                device="cuda",
                                dtype=torch.float32)

    cur_trans_matrix = torch.tensor(t_reg, device="cuda", dtype=torch.float32)

    pts_set_aff_np = (torch.mm(pts_set_src, cur_aff_left) +
                      cur_trans_matrix).contiguous().detach().cpu().numpy()

    return pts_set_aff_np


def do_xy_transform(pts_set_src, dx, dy):
    cur_aff_left = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda")
    cur_trans_matrix = torch.tensor([dx, dy], device="cuda")

    pts_set_aff_torch = (torch.mm(pts_set_src, cur_aff_left) +
                         cur_trans_matrix)
    pts_set_aff_np = pts_set_aff_torch.contiguous().detach().cpu().numpy()

    return pts_set_aff_np


def find_max_contour_box(img):
    # find contours in the thresholded image
    # Check OpenCV version
    cv2_major = cv2.__version__.split('.')[0]
    if cv2_major == '3':
        _, im_contours, im_hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)
    else:
        im_contours, im_hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if (len(im_contours) == 0):
        return None, None

    max_area = 0
    max_area_idx = 0
    max_bounding_rect = None
    for i in range(len(im_contours)):
        bounding_rect = cv2.boundingRect(im_contours[i])
        cur_area = cv2.contourArea(im_contours[i])
        if (cur_area > max_area):
            max_area = cur_area
            max_area_idx = i
            max_bounding_rect = bounding_rect

    return im_contours[max_area_idx], max_bounding_rect


def flood_fill(np_mask, canvas_size=(224, 224)):
    np_filld_copy = np_mask.copy()
    zero_filld = np.zeros((canvas_size[0] + 2, canvas_size[1] + 2), np.uint8)

    cv2.floodFill(np_filld_copy, zero_filld, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(np_filld_copy)
    im_fill_out = np_mask | im_floodfill_inv

    return im_fill_out


def get_group(img_fp,
              gt_rsz=(224, 224),
              seg_max_dist=20,
              seg_ratio=0.5,
              seg_kernel_size=3,
              seg_sigma=0,
              rag_sigma=255.0,
              rag_connectivity=2,
              rag_mode="distance",
              mer_thresh=30,
              is_merge=True
              ):

    target = PIL.Image.open(img_fp)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image

    target = target.convert("RGB").resize(gt_rsz, PIL.Image.Resampling.BICUBIC)
    img = np.array(target).astype("float")

    seg_quickshift = segmentation.quickshift(
        img, ratio=seg_ratio, kernel_size=seg_kernel_size, max_dist=seg_max_dist, sigma=seg_sigma)

    seg = seg_quickshift
    if (is_merge):
        g = graph.rag_mean_color(
            img,
            seg,
            connectivity=rag_connectivity,
            mode=rag_mode,
            sigma=rag_sigma,
        )
        seg = graph.merge_hierarchical(seg,
                                       g,
                                       thresh=mer_thresh,
                                       rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=_merge_mean_color,
                                       weight_func=_weight_mean_color)

    nb_layers = None
    if len(seg.shape) == 2:
        if nb_layers is None:
            nb_layers = seg.max() + 1
        masks = np.zeros((seg.shape[0], seg.shape[1], nb_layers)).astype(bool)
        m = masks.reshape((-1, nb_layers))
        s = seg.reshape((-1, ))
        m[np.arange(len(m)), s] = 1
        assert np.all(masks.argmax(axis=2) == seg)
    else:
        masks = seg

    cond_masks = []
    # add an alpha channel to img (224, 224, 4)
    img = np.dstack((img, np.ones((img.shape[0], img.shape[1])) * 255))
    for layer in range(masks.shape[2]):
        mask = masks[:, :, layer]

        mask_repeat = np.expand_dims(mask, 2).repeat(4, axis=2)
        mask_img_rgba = img * mask_repeat

        mask_alpha = mask_img_rgba[:, :, 3]  # 255
        mask_alpha = np.ascontiguousarray(mask_alpha, dtype=np.uint8)

        cond_masks.append(mask)

    # stack all masks
    cond_masks = np.stack(cond_masks, axis=2)

    return img, cond_masks
# -------------------------------------------------


def toposort_path(cur_tar_img_svg_path_info):
    tpsort = graphlib.TopologicalSorter()

    for pzi in range(len(cur_tar_img_svg_path_info) - 1):
        p_info_pzi = cur_tar_img_svg_path_info[pzi]
        pzi_img_mask_rgba = PIL.Image.open(
            p_info_pzi["tar_svg_path_mask_sub_fp"])
        pzi_img_mask_rgba = np.array(pzi_img_mask_rgba)
        pzi_img_mask = (pzi_img_mask_rgba[:, :, 3] > 0)

        for pzj in range(pzi + 1, len(cur_tar_img_svg_path_info)):
            p_info_pzj = cur_tar_img_svg_path_info[pzj]
            pzj_img_mask_rgba = PIL.Image.open(
                p_info_pzj["tar_svg_path_mask_sub_fp"])
            pzj_img_mask_rgba = np.array(pzj_img_mask_rgba)
            # pzj_img_mask = (pzj_img_mask_rgba[:, :, 3] == 255)
            pzj_img_mask = (pzj_img_mask_rgba[:, :, 3] > 0)

            overlap_mask = pzi_img_mask * pzj_img_mask
            sum_overlap_mask = overlap_mask.sum()

            if (sum_overlap_mask == 0):
                continue

            area_prop_pzi = sum_overlap_mask * 1.0 / pzi_img_mask.sum()
            area_prop_pzj = sum_overlap_mask * 1.0 / pzj_img_mask.sum()

            if (area_prop_pzi > area_prop_pzj):
                tpsort.add(pzi, pzj)
            else:
                tpsort.add(pzj, pzi)

    tpsort_stk = [*tpsort.static_order()]

    new_rank = []
    toposort_cur_tar_img_svg_path_info = []

    tp_cnt = 0
    for i in range(len(cur_tar_img_svg_path_info)):
        if (i in tpsort_stk):
            toposort_cur_tar_img_svg_path_info.append(
                cur_tar_img_svg_path_info[tpsort_stk[tp_cnt]])
            new_rank.append(tpsort_stk[tp_cnt])
            tp_cnt += 1
        else:
            toposort_cur_tar_img_svg_path_info.append(
                cur_tar_img_svg_path_info[i])
            new_rank.append(i)

    assert (tp_cnt == len(tpsort_stk))
    cur_tar_img_svg_path_info = toposort_cur_tar_img_svg_path_info

    # ------------------------------
    cur_tar_shapes_ini = []
    cur_tar_shapes_ini_groups = []

    for pz in range(len(toposort_cur_tar_img_svg_path_info)):
        p_info = toposort_cur_tar_img_svg_path_info[pz]
        svg_aff_path_fp = p_info["tar_svg_path_sub_fp"]

        cur_w, cur_h, cur_shapes, cur_shape_groups = pydiffvg.svg_to_scene(
            svg_aff_path_fp)

        p_cnt = 0
        cur_path = None
        for path in cur_shapes:
            cur_path = path
            p_cnt += 1

        assert (p_cnt == 1)

        cur_tar_shapes_ini.append(cur_path)
        fill_color_init = p_info["fill_color_target"]

        if "cur_path_ini_color" in p_info:
            fill_color_init = p_info["cur_path_ini_color"]
        # ---------------------------------------------

        cur_path_group = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor(
            [pz]),
            fill_color=fill_color_init,
            use_even_odd_rule=False)
        cur_tar_shapes_ini_groups.append(cur_path_group)

    return toposort_cur_tar_img_svg_path_info, cur_tar_shapes_ini, cur_tar_shapes_ini_groups
# -------------------------------------------------
