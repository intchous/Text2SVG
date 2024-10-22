import random
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import torch
import pydiffvg

from deepsvg.my_svg_dataset_pts import cubic_segments_to_points


def load_model1(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location=torch.device('cpu'))


def load_model2(checkpoint_path, model):
    state = torch.load(checkpoint_path)

    model.load_state_dict(state, strict=False)


def random_select_files(file_list, num_samples=50000):
    r0_files = filter_ending_files(file_list)
    return random.sample(r0_files, num_samples)


def filter_ending_files(file_list, ending="r0"):
    fn_list = []
    for fn in file_list:
        fn_pre = fn.split(".")[0]
        if (fn_pre.endswith(ending)):
            fn_list.append(fn)
    return fn_list


def imscatter(x, y, image, ax=None, zoom=1, offset=(0, 0)):
    """
    Function to plot image on specified x, y coordinates with an offset.
    """
    if ax is None:
        ax = plt.gca()
    im = offsetbox.OffsetImage(image, zoom=zoom, cmap='gray')
    ab = offsetbox.AnnotationBbox(
        im, (x + offset[0], y + offset[1]), frameon=False, pad=0.0)
    ax.add_artist(ab)


def tensor_to_img(tensor,  to_grayscale=True):
    """
    Convert a tensor in the shape [C, H, W] to a numpy image in the shape [H, W, C] with uint8 type.
    """
    tensor = tensor.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

    # Convert to grayscale if required
    if to_grayscale and tensor.shape[-1] == 3:
        tensor = tensor.mean(dim=-1, keepdim=True)

    # Convert to uint8
    tensor = (tensor * 255).clamp(0, 255).byte()

    img = tensor.cpu().numpy()

    return img


# -----------------------------------------------------------
def pts_to_pathObj(convert_points):
    # Number of control points per segment is 2. Hence, calculate the total number of segments.
    num_segments = int(convert_points.shape[0] / 3)
    num_control_points = [2] * num_segments
    num_control_points = torch.LongTensor(num_control_points)

    # Create a path object
    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=convert_points,
        stroke_width=torch.tensor(0.0),
        is_closed=True
    )

    return path


# 注意: 这里的affine_transform是以坐标原点为中心的
def apply_affine_transform_origin(points, theta, tx, ty, s):
    """
    Apply affine transformation, including rotation, translation, and overall scaling.
    """
    device = points.device

    # Create the affine transformation matrix for rotation and scaling.
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    aff_left = torch.stack([
        s * cos_theta, -s * sin_theta,
        s * sin_theta, s * cos_theta
    ]).view(2, 2).to(device)

    transformed_points = torch.mm(points, aff_left)

    # Add the translation.
    transformed_points += torch.stack([tx, ty]).to(device)

    return transformed_points


def apply_affine_transform(points, theta, tx, ty, s):
    # 以points中心进行affine
    """
    Apply affine transformation, including rotation, translation, and overall scaling.
    """
    device = points.device

    center = torch.mean(points, dim=0)

    # Translate points to center around the origin
    points_centered = points - center

    # Apply affine transformation (rotation and scaling)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    aff_left = torch.stack([
        s * cos_theta, -s * sin_theta,
        s * sin_theta, s * cos_theta
    ]).view(2, 2).to(device)

    transformed_points = torch.mm(points_centered, aff_left)

    # Translate back to the original location and apply overall translation
    translation = torch.stack([tx, ty])
    transformed_points += center + translation

    return transformed_points.to(device)


def cubics_to_points_affine(cubics, theta, tx, ty, s, s_norm, h=224, w=224, use_affine_norm=False):
    convert_points_ini = cubic_segments_to_points(cubics)

    # Apply affine transform to convert_points
    convert_points = apply_affine_transform(
        convert_points_ini, theta, tx, ty, s)

    # Clamp the values of convert_points_ini to be in [0,1]
    if (use_affine_norm):
        # convert_points_ini = torch.clamp(convert_points_ini, 0, 1)
        # convert_points = torch.clamp(convert_points, 0, 1)
        convert_points_ini_trans = s_norm.inverse_transform(convert_points_ini)
        convert_points_trans = s_norm.inverse_transform(convert_points)

        return convert_points_trans, convert_points_ini_trans

    else:
        # convert_points_ini = torch.clamp(convert_points_ini, 0, 1*h)
        # convert_points = torch.clamp(convert_points, 0, 1*h)

        return convert_points, convert_points_ini


def z_to_affine_pts(z, theta, tx, ty, s, model, s_norm, h=224, w=224, use_affine_norm=False):
    # 使用z生成点序列 (每生成一条path都调用1次, 太耗时了)
    generated_data = model(
        args_enc=None, args_dec=None, z=z.unsqueeze(1).unsqueeze(2))
    generated_pts = generated_data["args_logits"]

    recon_data_output = generated_pts.squeeze(1)

    bat_s = 1
    ini_cubics_batch = recon_data_output.view(
        bat_s, -1, 4, 2)
    ini_cubics = ini_cubics_batch[0]
    # ini_cubics.shape:  torch.Size([10, 4, 2])

    if (use_affine_norm):
        ini_cubics_trans = ini_cubics
    else:
        ini_cubics_trans = s_norm.inverse_transform(ini_cubics)

    convert_points, convert_points_ini = cubics_to_points_affine(
        cubics=ini_cubics_trans, theta=theta, tx=tx, ty=ty, s=s, s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

    return convert_points, convert_points_ini


def recon_to_affine_pts(recon_data_output, theta, tx, ty, s, s_norm, h=224, w=224, use_affine_norm=False):

    bat_s = 1
    ini_cubics_batch = recon_data_output.view(
        bat_s, -1, 4, 2)
    ini_cubics = ini_cubics_batch[0]
    # ini_cubics.shape:  torch.Size([10, 4, 2])

    if (use_affine_norm):
        ini_cubics_trans = ini_cubics
    else:
        ini_cubics_trans = s_norm.inverse_transform(ini_cubics)

    convert_points, convert_points_ini = cubics_to_points_affine(
        cubics=ini_cubics_trans, theta=theta, tx=tx, ty=ty, s=s, s_norm=s_norm, h=h, w=w, use_affine_norm=use_affine_norm)

    return convert_points, convert_points_ini


def cubics_to_pathObj_affine(cubics, theta, tx, ty, s, s_norm, h=224, w=224, use_affine_norm=False):
    """Given a cubics tensor, return a pathObj."""
    convert_points, convert_points_ini = cubics_to_points_affine(
        cubics, theta, tx, ty, s, s_norm, h, w, use_affine_norm)

    path = pts_to_pathObj(convert_points)

    return path, convert_points_ini


# ----------------------------------
def paths_to_shapes(path_list, fill_color_list, stroke_width_list=None, stroke_color_list=None):
    if stroke_width_list is not None:
        for i, path in enumerate(path_list):
            path.stroke_width = stroke_width_list[i]
    
    tp_shapes = path_list
    tp_shape_groups = [
        pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([i]),
            fill_color=color,
            stroke_color=None if stroke_color_list is None else stroke_color_list[i],
            use_even_odd_rule=False
        ) for i, color in enumerate(fill_color_list)
    ]

    return tp_shapes, tp_shape_groups


def save_paths_svg(path_list,
                   fill_color_list=[],
                   stroke_width_list=[],
                   stroke_color_list=[],
                   svg_path_fp="",
                   canvas_height=224,
                   canvas_width=224):

    tp_shapes = []
    tp_shape_groups = []

    for pi in range(len(path_list)):
        ini_path = path_list[pi]
        ini_path.stroke_width = stroke_width_list[pi]
        tp_shapes.append(ini_path)

        # fill_color=torch.FloatTensor([0.5, 0.5, 0.5, 1.0])
        tp_fill_color = fill_color_list[pi]

        tp_path_group = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor([pi]),
                                            fill_color=tp_fill_color,
                                            stroke_color=stroke_color_list[pi],
                                            use_even_odd_rule=False)
        tp_shape_groups.append(tp_path_group)

    if (len(svg_path_fp) > 0):
        pydiffvg.save_svg(svg_path_fp, canvas_width, canvas_height, tp_shapes,
                          tp_shape_groups)

    return tp_shapes, tp_shape_groups


def render_and_compose(tmp_paths_list, color_list, stroke_width_list=None, stroke_color_list=None, w=224, h=224, svg_path_fp="", para_bg=None, render_func=None, return_shapes=False, device="cuda"):
    if para_bg is None:
        para_bg = torch.tensor(
            [1., 1., 1.], requires_grad=False, device=device)
    if (render_func is None):
        render_func = pydiffvg.RenderFunction.apply

    tp_shapes, tp_shape_groups = save_paths_svg(
        path_list=tmp_paths_list, fill_color_list=color_list, stroke_width_list=stroke_width_list, stroke_color_list=stroke_color_list, svg_path_fp=svg_path_fp, canvas_height=h, canvas_width=w)

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        w, h, tp_shapes, tp_shape_groups)
    tmp_img = render_func(w, h, 2, 2, 0, None, *scene_args)

    # Compose img with white background
    combined_img = tmp_img[:, :, 3:4] * tmp_img[:,
                                                :, :3] + para_bg * (1 - tmp_img[:, :, 3:4])
    recon_imgs = combined_img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW

    if (return_shapes):
        return recon_imgs, combined_img, tp_shapes, tp_shape_groups

    # return recon_imgs, tmp_img
    return recon_imgs, combined_img


def render_and_blend(tmp_paths_list, color_list, w=224, h=224, svg_path_fp="", para_bg=None, render_func=None, device="cuda"):
    if para_bg is None:
        para_bg = torch.tensor(
            [1., 1., 1.], requires_grad=False, device=device)
    if (render_func is None):
        render_func = pydiffvg.RenderFunction.apply

    if len(tmp_paths_list) == 0:
        return None

    # 预先分配内存，存储所有渲染的路径图像
    all_images = torch.zeros(len(tmp_paths_list), 4, h, w, device=device)

    # 渲染每个路径，并存储到all_images中
    for idx, (path, color) in enumerate(zip(tmp_paths_list, color_list)):
        tp_shapes, tp_shape_groups = save_paths_svg(
            path_list=[path], fill_color_list=[color], svg_path_fp=svg_path_fp, canvas_height=h, canvas_width=w)

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            w, h, tp_shapes, tp_shape_groups)
        img = render_func(w, h, 2, 2, 0, None, *scene_args)
        img = img.permute(2, 0, 1)  # HWC -> CHW
        all_images[idx] = img

    # 按照composite函数的逻辑进行合成
    n = len(tmp_paths_list)
    alpha = (1 - all_images[n - 1, 3:4])
    rgb = all_images[n - 1, :3] * all_images[n - 1, 3:4]
    for i in reversed(range(n-1)):
        alpha_comp = (1 - all_images[i, 3:4])
        rgb = rgb + all_images[i, :3] * all_images[i, 3:4] * alpha
        alpha = alpha * alpha_comp

    # Composite with the background
    combined_rgb = rgb + alpha * para_bg.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    # rgb.shape:  torch.Size([3, 224, 224])
    # combined_rgb.shape:  torch.Size([1, 3, 224, 224])

    # Combined image is in NCHW format
    return combined_rgb, combined_rgb.permute(0, 2, 3, 1).squeeze(0)


# ----------------------------------------------------
def regularization_loss(z):
    mean = z.mean()
    std = z.std()
    return mean**2 + (std - 1)**2


def l2_regularization_loss(z):
    return torch.norm(z, p=2) ** 2


def kl_divergence_loss(z):
    mean = z.mean()
    variance = z.var()  # Directly compute variance
    kl_loss = -0.5 * (1 + torch.log(variance) - mean**2 - variance)
    return kl_loss.sum()


def safe_pow(t, exponent, eps=1e-6):
    return t.clamp(min=eps).pow(exponent)


def opacity_penalty(colors, coarse_learning=True):
    factor = 1 if coarse_learning else 0
    alpha = colors[:, 3]  # 假设alpha在颜色张量的最后一维
    if coarse_learning:
        penalty = factor * safe_pow(alpha, 0.5).mean()
    else:
        binary_alpha = (alpha > 0.5).float()
        penalty = factor * safe_pow(binary_alpha, 0.5).mean()
    return penalty


def binary_alpha_penalty_sigmoid(colors):
    alpha = colors[:, 3]
    binary_alpha = torch.sigmoid(10 * (alpha - 0.5))
    loss = torch.mean((binary_alpha - alpha) ** 2)
    return loss


def binary_alpha_penalty_l1(colors):
    alpha = colors[:, 3]
    loss = torch.mean(torch.min(alpha, 1 - alpha))  # 使用L1损失
    return loss


def l1_penalty(tensor):
    return torch.sum(torch.abs(tensor))


def control_polygon_distance(points):
    # Calculate squared differences between successive points
    diff = points[1:] - points[:-1]
    # Compute the squared distances (i.e., sum of squared differences along the last dimension)
    squared_distances = (diff ** 2).sum(dim=1)
    # Return the mean of the squared distances
    return squared_distances.mean()

# ----------------------------------------------------
