import os
import numpy as np
import random
import math

import PIL
import PIL.Image
from PIL import Image

import torch
from torchvision import transforms

import cairosvg
import pydiffvg
from deepsvg.svglib.svg import SVG


def get_bezier_circle(radius=1, segments=4, bias=None):
    points = []
    if bias is None:
        bias = (random.random(), random.random())
    avg_degree = 360 / (segments * 3)
    for i in range(0, segments * 3):
        point = (np.cos(np.deg2rad(i * avg_degree)),
                 np.sin(np.deg2rad(i * avg_degree)))
        points.append(point)
    points = torch.tensor(points)
    points = (points) * radius + torch.tensor(bias).unsqueeze(dim=0)
    points = points.type(torch.FloatTensor)
    return points


def ycrcb_conversion(im, format='[bs x 3 x 2D]', reverse=False):
    mat = torch.FloatTensor([
        [65.481 / 255, 128.553 / 255,
         24.966 / 255],  # ranged_from [0, 219/255]
        [-37.797 / 255, -74.203 / 255,
         112.000 / 255],  # ranged_from [-112/255, 112/255]
        [112.000 / 255, -93.786 / 255,
         -18.214 / 255],  # ranged_from [-112/255, 112/255]
    ]).to(im.device)

    if reverse:
        mat = mat.inverse()

    if format == '[bs x 3 x 2D]':
        im = im.permute(0, 2, 3, 1)
        im = torch.matmul(im, mat.T)
        im = im.permute(0, 3, 1, 2).contiguous()
        return im
    elif format == '[2D x 3]':
        im = torch.matmul(im, mat.T)
        return im
    else:
        raise ValueError


def load_target(fp, size=(224, 224), return_rgb=True, device="cuda"):
    target = PIL.Image.open(fp)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    if target.size != size:
        target = target.resize(size, PIL.Image.Resampling.BICUBIC)

    if (return_rgb):
        return target
    else:
        transforms_ = []
        # transforms_.append(transforms.Resize(size, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.ToTensor())
        data_transforms = transforms.Compose(transforms_)  # w,h,c -> c,h,w
        gt = data_transforms(target).unsqueeze(0).to(device)
        return gt

# --------------------------------------------


def save_path_svg(ini_path,
                  fill_color=torch.FloatTensor([0.5, 0.5, 0.5, 1.0]),
                  svg_path_fp="",
                  canvas_height=224,
                  canvas_width=224):

    tp_shapes = []
    tp_shape_groups = []
    tp_shapes.append(ini_path)
    tp_fill_color = fill_color

    tp_path_group = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor([0]),
                                        fill_color=tp_fill_color,
                                        use_even_odd_rule=False)
    tp_shape_groups.append(tp_path_group)

    if (len(svg_path_fp) > 0):
        pydiffvg.save_svg(svg_path_fp, canvas_width, canvas_height, tp_shapes,
                          tp_shape_groups)

    return tp_shapes, tp_shape_groups


def path2img(ini_path, fill_color_init, h, w, svg_path_fp=""):

    cur_shapes, cur_shape_groups = save_path_svg(ini_path,
                                                 fill_color=fill_color_init,
                                                 svg_path_fp=svg_path_fp,
                                                 canvas_height=h,
                                                 canvas_width=w)

    empty_flag = False
    for c_path in cur_shapes:
        if (c_path.points.shape[0] < 3):
            empty_flag = True
        break

    if (empty_flag):
        img = torch.zeros([h, w, 4]).to("cuda")
    else:
        canvas_width = w
        canvas_height = h
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, cur_shapes, cur_shape_groups)

        render = pydiffvg.RenderFunction.apply
        img = render(
            canvas_width,  # width
            canvas_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,
            *scene_args)
        # print("img.shape = ", img.shape)  # (224, 224, 4)

    return img


def do_affine_transform(pts_set_src, s_reg, R_reg, t_reg):
    pts_device = pts_set_src.device
    cur_aff_left = torch.tensor(s_reg * R_reg,
                                device=pts_device,
                                dtype=torch.float32)

    cur_trans_matrix = torch.tensor(
        t_reg, device=pts_device, dtype=torch.float32)

    pts_set_aff = (torch.mm(pts_set_src, cur_aff_left) +
                   cur_trans_matrix).contiguous()

    return pts_set_aff


def move_bbox_center_to_img_center(pts_set_src, img_hw):
    center_h, center_w = img_hw[0] / 2.0, img_hw[1] / 2.0

    bbox_min = torch.min(pts_set_src, dim=0).values
    bbox_max = torch.max(pts_set_src, dim=0).values
    bbox_center = (bbox_max + bbox_min) / 2.0

    center_pts = bbox_center

    translate_dist = torch.tensor(
        [center_h, center_w], dtype=torch.float32) - center_pts

    predicted_max_coords = bbox_max + translate_dist
    predicted_min_coords = bbox_min + translate_dist

    max_values = torch.tensor([img_hw[0], img_hw[1]])
    min_values = torch.tensor([0.0, 0.0])

    if torch.any(predicted_max_coords > max_values):
        offset = torch.max(predicted_max_coords - max_values,
                           torch.tensor([0.0, 0.0]))
        translate_dist -= offset

    if torch.any(predicted_min_coords < min_values):
        offset = torch.max(min_values - predicted_min_coords,
                           torch.tensor([0.0, 0.0]))
        translate_dist += offset

    s_reg1 = 1  # scaling factor
    R_reg1 = torch.eye(2)  # identity matrix for rotation
    t_reg1 = translate_dist  # vector for translation
    translate_pts = do_affine_transform(
        pts_set_src, s_reg=s_reg1, R_reg=R_reg1, t_reg=t_reg1)

    return translate_pts


def scale_bbox_to_target_size(pts_set_src, img_hw, target_scale=1.0):

    bbox_min = torch.min(pts_set_src, dim=0).values
    bbox_max = torch.max(pts_set_src, dim=0).values

    bbox_size = bbox_max - bbox_min
    target_size = torch.tensor(
        [img_hw[0] * target_scale, img_hw[1] * target_scale])

    scale_fac = torch.min(target_size / bbox_size)

    s_reg2 = scale_fac  # scaling factor
    R_reg2 = torch.eye(2)  # identity matrix for rotation

    # t_reg2 = torch.tensor([center_h, center_w]) * (1 - scale_fac)
    t_reg2 = (1 - scale_fac) * (bbox_max + bbox_min) / 2.0

    scale_pts = do_affine_transform(
        pts_set_src, s_reg=s_reg2, R_reg=R_reg2, t_reg=t_reg2)

    return scale_pts


def rotate_pts_func(pts_set_src, theta_degrees, rotation_center):
    """
    This function rotates the point set around the specified rotation center.

    :param pts_set_src: The original points set to be rotated.
    :param theta_degrees: The angle of rotation in degrees.
    :param rotation_center: The center [x, y] around which to rotate.
    :return: Rotated point set.
    """
    # Convert rotation angle to radians.
    theta = math.radians(theta_degrees)

    # Define the rotation matrix.
    R_reg3 = torch.tensor([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta), math.cos(theta)]])

    # Translation vector for rotation around rotation_center.
    # t_reg3 = torch.tensor([0.0, 0.0])
    t_reg3 = rotation_center - torch.matmul(rotation_center, R_reg3)

    # Scaling factor remains unchanged during rotation.
    s_reg3 = 1

    # Apply rotation.
    rotated_pts = do_affine_transform(
        pts_set_src, s_reg=s_reg3, R_reg=R_reg3, t_reg=t_reg3)

    return rotated_pts


def move_scale_bbox_pts(pts_set_src, img_hw, target_scale=0.9):
    translate_pts = move_bbox_center_to_img_center(
        pts_set_src=pts_set_src, img_hw=img_hw)

    scale_pts = scale_bbox_to_target_size(
        pts_set_src=translate_pts, img_hw=img_hw, target_scale=target_scale)

    return scale_pts


def move_scale_rotate_pts(pts_set_src, img_hw=(224, 224), target_scale=0.9, theta_degrees=0):
    """
    This function first translates the point set to the center of the image,
    then scales it, and finally rotates it around the center of its bounding box.

    :param pts_set_src: The original points set.
    :param img_hw: The dimensions of the image as [height, width].
    :param target_scale: Target scale for bounding box resizing.
    :param theta_degrees: The angle of rotation in degrees.
    :return: Transformed point set.
    """

    translate_pts = move_bbox_center_to_img_center(
        pts_set_src=pts_set_src, img_hw=img_hw)

    scale_pts = scale_bbox_to_target_size(
        pts_set_src=translate_pts, img_hw=img_hw, target_scale=target_scale)

    # Calculate bounding box center for rotation center
    bbox_min = torch.min(scale_pts, dim=0).values
    bbox_max = torch.max(scale_pts, dim=0).values
    rotation_center = (bbox_max + bbox_min) / 2.0

    # Step 3: Rotation around bounding box center.
    rotated_pts = rotate_pts_func(
        pts_set_src=scale_pts, theta_degrees=theta_degrees, rotation_center=rotation_center)

    return rotated_pts


# --------------------------------------------
def load_init_svg(svg_fp,
                  canvas_size=(224, 224),
                  experiment_dir="./svg_ref_img/",
                  scale_fac=1.334,
                  svg_cario_dir="./svg_ref_cairo/",
                  save_cairo_img=False,
                  save_diffvg_img=False,
                  requires_grad=False,
                  use_cuda=False,
                  trainable_stroke=False,
                  add_circle=False):

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(svg_cario_dir, exist_ok=True)

    shapes = []
    shape_groups = []

    infile = svg_fp
    im_fn = infile.split('/')[-1]
    im_pre, im_ext = os.path.splitext(im_fn)

    if (save_cairo_img):
        fp_cairosvg_img = os.path.join(experiment_dir, im_pre + "_cairo.png")
        cairosvg.svg2png(url=infile,
                         write_to=fp_cairosvg_img,
                         output_width=canvas_size[0],
                         output_height=canvas_size[1])

    fp_cairosvg_svg = os.path.join(svg_cario_dir, im_pre + "_cairo.svg")
    cairosvg.svg2svg(url=infile,
                     write_to=fp_cairosvg_svg,
                     output_width=(canvas_size[0] * scale_fac),
                     output_height=(canvas_size[1] * scale_fac))

    # ------------------------------------------
    infile = fp_cairosvg_svg
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        infile)
    canvas_width = canvas_size[0]
    canvas_height = canvas_size[1]

    outfile = os.path.join(svg_cario_dir, im_pre + ".svg")
    pydiffvg.save_svg(outfile, canvas_width,
                      canvas_height, shapes, shape_groups)

    infile = outfile
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        infile)

    assert (len(shapes) == len(shape_groups))
    # ------------------------------------------

    # ------------------------------------------
    if (add_circle):
        num_control_points = [2] * 4
        msk_x_mean = 15
        msk_y_mean = 15
        circle_center = (msk_x_mean, msk_y_mean)
        circle_points = get_bezier_circle(radius=5,
                                          segments=4,
                                          bias=circle_center)

        circle_path = pydiffvg.Path(
            num_control_points=torch.LongTensor(num_control_points),
            points=circle_points,
            stroke_width=torch.tensor(0.0),
            is_closed=True)
        shapes.append(circle_path)
        fill_color_init = torch.FloatTensor([0.5, 0.5, 0.5, 1.0])
        circle_path_group = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor(
            [len(shapes) - 1]),
            fill_color=fill_color_init,
            use_even_odd_rule=False)
        shape_groups.append(circle_path_group)
        pydiffvg.save_svg(outfile, canvas_size[0], canvas_size[1], shapes,
                          shape_groups)
    # -------------------------------

    if (save_diffvg_img):
        diffvg_width = canvas_size[0]
        diffvg_height = canvas_size[1]

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            diffvg_width, diffvg_height, shapes, shape_groups)

        render = pydiffvg.RenderFunction.apply
        img = render(
            diffvg_width,  # width
            diffvg_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,
            *scene_args)

        # Transform to gamma space
        # new_img_fn_pydiffvg = experiment_dir + im_pre + '_init.png'
        new_img_fn_pydiffvg = os.path.join(experiment_dir, im_pre + '.png')
        pydiffvg.imwrite(img.cpu(), new_img_fn_pydiffvg, gamma=1.0)

    # delete cairosvg files
    # os.remove(fp_cairosvg_img)
    # os.remove(fp_cairosvg_svg)

    point_var = []
    color_var = []
    for path in shapes:
        if (use_cuda):
            path.points = path.points.to("cuda")
        path.points.requires_grad = requires_grad
        point_var.append(path.points)
    for group in shape_groups:
        if (group.fill_color is None):
            group.fill_color = torch.FloatTensor([1.0, 1.0, 1.0, 0.0])
        if (use_cuda):
            group.fill_color = group.fill_color.to("cuda")
        group.fill_color.requires_grad = requires_grad
        color_var.append(group.fill_color)

    if trainable_stroke:
        stroke_width_var = []
        stroke_color_var = []
        for path in shapes:
            if (use_cuda):
                path.stroke_width = path.stroke_width.to("cuda")
            path.stroke_width.requires_grad = requires_grad
            stroke_width_var.append(path.stroke_width)
        for group in shape_groups:
            if (use_cuda):
                group.stroke_color = group.stroke_color.to("cuda")
            group.stroke_color.requires_grad = requires_grad
            stroke_color_var.append(group.stroke_color)
        return shapes, shape_groups, point_var, color_var, stroke_width_var, stroke_color_var
    else:
        return shapes, shape_groups, point_var, color_var


def inipts_to_validpts_norm(pts_norm, threshold=0.05):
    """
    Trims points starting from the first point that is close to [0, 0].

    Args:
    - pts (torch.Tensor): The points tensor of shape [N, 2].
    - threshold (float): Points with both x and y coordinates below this threshold are considered close to [0, 0].

    Returns:
    - torch.Tensor: A tensor containing only the valid points before the first close to [0, 0] point.
    """

    # Check if all values in pts_norm are between 0 and 1
    if not ((pts_norm >= 0).all() and (pts_norm <= 1).all()):
        raise ValueError(
            "All values in pts_norm should be in the range [0, 1].")

    # Create a boolean mask where True indicates points with both coordinates below the threshold
    mask = (pts_norm < threshold).all(dim=1)

    # If no point is close to [0,0], return the input tensor
    if not mask.any():
        return pts_norm
    else:
        # Find the first occurrence of a point close to [0,0]
        effective_len = mask.nonzero(as_tuple=True)[0][0]
        # Return the tensor up to the effective length, but not including the close-to-zero point
        return pts_norm[:effective_len]


def inipts_to_validpts(data_pts_trans, w, eos_threshold=0.05):
    # # Find the position of the first occurrence of -1 in each sequence
    # end_indx = torch.argmax(data_pts_trans == -1, dim=1)

    # # Determine the effective sequence length for each sequence
    # if(end_indx > 0):
    #     effective_len = end_indx
    # else:
    #     effective_len = data_pts_trans.size(0)

    # Create a boolean tensor of the same shape as data_pts_trans
    # Mark the positions where the value is less than -0.99
    # mask = data_pts_trans < (-0.9 * w)
    mask = data_pts_trans < (eos_threshold * w)

    # Find the positions where both numbers are less than -0.99
    mask = mask.all(dim=1)

    # If there is no such position, return the length of the tensor
    if not mask.any():
        effective_len = data_pts_trans.size(0)
    else:
        # Otherwise, find the first position where both numbers are less than -0.99
        effective_len = mask.nonzero(as_tuple=True)[0][0]

    # effective_len = int(effective_len)
    # print("effective_len: ", effective_len)

    # Truncate to effective sequence length
    # data_pts_trans_new = data_pts_trans[:effective_len].contiguous()
    data_pts_trans_new = data_pts_trans[:effective_len]

    return data_pts_trans_new


def pts_to_path(data_pts_trans, tmp_svg_path_fp, w, h, eos_threshold=0.05, return_path=False, return_points=False):

    new_points = inipts_to_validpts(
        data_pts_trans=data_pts_trans, w=w, eos_threshold=eos_threshold)
    # num_seg = math.ceil((new_points.shape[0]-1)/3.0)
    # new_num_control_points = [2]*num_seg
    new_num_control_points = [2] * int(new_points.shape[0] / 3)

    # new_points = torch.tensor(new_points, dtype=torch.float32)
    new_num_control_points = torch.LongTensor(new_num_control_points)
    tmp_path = pydiffvg.Path(
        num_control_points=new_num_control_points,
        points=new_points,
        stroke_width=torch.tensor(0.0),
        is_closed=True)

    if (return_path):
        return tmp_path, new_points

    fill_color_tmp = torch.FloatTensor([0.0, 0.0, 0.0, 1.0])

    cur_path_img = path2img(ini_path=tmp_path, fill_color_init=fill_color_tmp,
                            h=h, w=w, svg_path_fp=tmp_svg_path_fp)

    if (return_points):
        return cur_path_img, new_points

    return cur_path_img


def pts_to_path_dict(data_pts_trans, tmp_svg_path_fp, w, h, eos_threshold=0.05, return_path=False, return_points=False):

    new_points = inipts_to_validpts(
        data_pts_trans=data_pts_trans, w=w, eos_threshold=eos_threshold)

    # Check if new_points is empty
    if new_points.shape[0] < 4:
        return {"valid_path": False}

    new_num_control_points = [2] * int(new_points.shape[0] / 3)

    new_num_control_points = torch.LongTensor(new_num_control_points)
    tmp_path = pydiffvg.Path(
        num_control_points=new_num_control_points,
        points=new_points,
        stroke_width=torch.tensor(0.0),
        is_closed=True)

    if (return_path):
        return {"valid_path": True, "path": tmp_path, "points": new_points}

    fill_color_tmp = torch.FloatTensor([0.0, 0.0, 0.0, 1.0])

    cur_path_img = path2img(ini_path=tmp_path, fill_color_init=fill_color_tmp,
                            h=h, w=w, svg_path_fp=tmp_svg_path_fp)

    if (return_points):
        return {"valid_path": True, "image": cur_path_img, "points": new_points}

    return {"valid_path": True, "image": cur_path_img}


def pts_to_path_with_length(data_pts_trans, tmp_svg_path_fp, w, h, length, return_path=False):
    effective_len = length - 1

    data_pts_trans_new = data_pts_trans[:effective_len]

    new_points = data_pts_trans_new
    new_num_control_points = [2] * int(new_points.shape[0] / 3)

    new_num_control_points = torch.LongTensor(new_num_control_points)
    tmp_path = pydiffvg.Path(
        num_control_points=new_num_control_points,
        points=new_points,
        stroke_width=torch.tensor(0.0),
        is_closed=True)

    if (return_path):
        return tmp_path, new_points

    fill_color_tmp = torch.FloatTensor([0.0, 0.0, 0.0, 1.0])

    cur_path_img = path2img(ini_path=tmp_path, fill_color_init=fill_color_tmp,
                            h=h, w=w, svg_path_fp=tmp_svg_path_fp)

    return cur_path_img


def imgs2grid(img_path_fp_list, save_grid_img_fp, rows=2, columns=5, img_w=224, img_h=224):
    # Create an empty grid to store the images
    img_w = int(img_w)
    img_h = int(img_h)
    grid_image = Image.new('RGBA', (columns * img_w, rows * img_h))
    len_img_path_fp_list = len(img_path_fp_list)

    for i in range(len_img_path_fp_list):
        tmp_img_path_fp = img_path_fp_list[i]
        # Open the image file
        tmp_img = Image.open(tmp_img_path_fp).resize((img_w, img_h))

        # Calculate the position in the grid
        tmp_row = i // columns
        tmp_col = i % columns

        # Calculate the coordinates for pasting the image
        paste_x = tmp_col * img_w
        paste_y = tmp_row * img_h

        # Paste the image into the grid
        grid_image.paste(tmp_img, (paste_x, paste_y))

    # Save the final grid image
    grid_image.save(save_grid_img_fp)


# ------------------------------------
# deepsvg preprocess
def simplify_svg(infile, output_cario_output_foler, output_folder, h=224, w=224, scale_fac=1.334, use_canonicalize=False, use_simplify_heuristic=False, if_simp_split=False, max_dist=8):
    im_fn = infile.split('/')[-1]
    im_pre, im_ext = os.path.splitext(im_fn)

    fp_cairosvg_svg = os.path.join(
        output_cario_output_foler, im_pre + "_cario.svg")
    cairosvg.svg2svg(url=infile,
                     write_to=fp_cairosvg_svg,
                     output_width=(w * scale_fac),
                     output_height=(h * scale_fac))

    tmp_svg = SVG.load_svg(fp_cairosvg_svg)

    if (use_canonicalize):
        tmp_svg = tmp_svg.canonicalize()

    if (use_simplify_heuristic):
        if (if_simp_split):
            # tmp_svg = tmp_svg.simplify_heuristic()
            tmp_svg = tmp_svg.split(max_dist=2, include_lines=False).simplify(
                tolerance=0.1, epsilon=0.2, angle_threshold=150., force_smooth=False).split(max_dist=max_dist)

        else:
            tmp_svg = tmp_svg.split(max_dist=2, include_lines=False).simplify(
                tolerance=0.1, epsilon=0.2, angle_threshold=150., force_smooth=False)

    len_groups = [path_group.total_len()
                  for path_group in tmp_svg.svg_path_groups]

    total_len = sum(len_groups)
    if total_len > 0:
        tmp_svg.save_svg(os.path.join(
            output_folder, f"{im_pre}_norm.svg"))

    # delete cairosvg files
    if (os.path.exists(fp_cairosvg_svg)):
        os.remove(fp_cairosvg_svg)

# ------------------------------------
