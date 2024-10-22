import torch
from torch.nn import functional as F
import kornia
from optim_utils.util_path import sample_points_by_length_distribution

dsample = kornia.geometry.transform.PyrDown()


def procrustes_distance(src_points, tgt_points):
    src_centered = src_points - src_points.mean(dim=0)
    tgt_centered = tgt_points - tgt_points.mean(dim=0)
    epsilon = 1e-8

    regularization = torch.eye(src_centered.size(1)).to(
        src_centered.device) * epsilon
    src_centered_t_tgt_centered = torch.matmul(
        src_centered.t(), tgt_centered) + regularization

    u, _, v = torch.svd(src_centered_t_tgt_centered)

    rotation = torch.matmul(u, v.t())

    src_rotated = torch.matmul(src_centered, rotation)

    scale = torch.sum(tgt_centered * src_rotated) / \
        (torch.sum(src_rotated * src_rotated) + epsilon)
    src_transformed = scale * src_rotated

    # Calculate the Procrustes distance with a small positive value for numerical stability
    distance = torch.sqrt(
        torch.sum((tgt_centered - src_transformed) ** 2) + epsilon)
    return distance


def local_procrustes_loss(src_points, tgt_points, window_size=5, return_avg=False):
    assert src_points.shape == tgt_points.shape, "src_points and tgt_points should have the same shape"
    assert window_size > 0, "window_size should be greater than 0"

    n_points = src_points.size(0)
    distance_sum = 0.0
    window_size = min(window_size, n_points)

    for i in range(n_points):
        src_window = src_points[torch.arange(i, i + window_size) % n_points]
        tgt_window = tgt_points[torch.arange(i, i + window_size) % n_points]
        distance_sum += procrustes_distance(src_window, tgt_window)

    if return_avg:
        return (distance_sum / n_points)
    else:
        return distance_sum


def local_procrustes_loss_centered(src_points, tgt_points, window_size=5, return_avg=False):
    assert src_points.shape == tgt_points.shape, "src_points and tgt_points should have the same shape"
    assert window_size > 0, "window_size should be greater than 0"

    n_points = src_points.size(0)
    distance_sum = 0.0
    window_size = min(window_size, n_points)
    half_window = window_size // 2

    for i in range(n_points):
        src_window = src_points[torch.arange(
            i - half_window, i + half_window + 1) % n_points]
        tgt_window = tgt_points[torch.arange(
            i - half_window, i + half_window + 1) % n_points]
        distance_sum += procrustes_distance(src_window, tgt_window)

    if return_avg:
        return (distance_sum / n_points)
    else:
        return distance_sum


def local_procrustes_loss_centeredv2(src_points, tgt_points, window_size=5, return_avg=False):
    assert src_points.shape == tgt_points.shape, "src_points and tgt_points should have the same shape"
    assert window_size > 0, "window_size should be greater than 0"

    n_points = src_points.size(0)
    window_size = min(window_size, n_points)
    half_window = window_size // 2

    indices = torch.arange(n_points + window_size - 1) % n_points
    windows = indices.unfold(0, window_size, 1)

    src_windows = src_points[windows]
    tgt_windows = tgt_points[windows]

    # Calculate Procrustes distance for each window
    distances = torch.stack([procrustes_distance(
        src_windows[i], tgt_windows[i]) for i in range(n_points)])

    if return_avg:
        return distances.mean()
    else:
        return distances.sum()


# ----------------------------------------------------------------
def laplacian_smoothing_loss(points, num_neighbors=1, weight=1.0):
    n_points = points.size(0)

    avg_neighbors = torch.zeros_like(points)

    for i in range(-num_neighbors, num_neighbors + 1):
        if i == 0:
            continue
        index_shift = (torch.arange(n_points) - i) % n_points
        avg_neighbors += points[index_shift]
    avg_neighbors /= (2 * num_neighbors)

    diff = points - avg_neighbors

    smoothness = torch.norm(diff, p=2)

    return weight * smoothness


def laplacian_smoothing_loss_with_curvature(points, num_neighbors=1, corner_threshold=0.2):
    n_points = points.size(0)

    avg_neighbors = torch.zeros_like(points)

    for i in range(-num_neighbors, num_neighbors + 1):
        if i == 0:
            continue
        index_shift = (torch.arange(n_points) - i) % n_points
        avg_neighbors += points[index_shift]
    avg_neighbors /= (2 * num_neighbors)

    diff = points - avg_neighbors

    # Calculate local curvature
    prev_points = torch.roll(points, shifts=1, dims=0)
    next_points = torch.roll(points, shifts=-1, dims=0)
    v1 = points - prev_points
    v2 = next_points - points
    curvature = 1 - torch.abs(torch.sum(v1 * v2, dim=-1) /
                              (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8))

    # Apply adaptive weighting based on curvature
    adaptive_weight = torch.where(curvature > corner_threshold, 0.1, 1.0)
    weighted_diff = diff * adaptive_weight.unsqueeze(-1)

    smoothness_loss = torch.norm(weighted_diff, p=2)

    return smoothness_loss
# ----------------------------------------


def kl_divergence(src_z):
    src_mean = torch.mean(src_z, dim=-1)
    src_std = torch.std(src_z, dim=-1)
    kl_div = 0.5 * torch.sum(src_std**2 + src_mean **
                             2 - 1 - torch.log(src_std**2), dim=-1)
    return kl_div.mean()


def svg_emd_loss(p_pred, p_target, p_target_sub=None, matching=None):

    n, m = len(p_pred), len(p_target)

    if n == 0:
        return 0.

    if p_target_sub is None or matching is None:
        p_target_sub, matching = sample_points_by_length_distribution(
            p_target, n, device=p_pred.device)

    # EMD - Vectorized reordering computation
    indices = torch.arange(n, device=p_pred.device).unsqueeze(0).repeat(n, 1)
    roll_indices = (indices + indices.T) % n
    reordered_ptarget_subs = torch.index_select(
        p_target_sub, 0, roll_indices.view(-1)).view(n, n, -1)

    # roll_indices.shape:  torch.Size([80, 80])
    # reordered_ptarget_subs.shape:  torch.Size([80, 80, 2])

    distances = torch.norm(p_pred.unsqueeze(
        0) - reordered_ptarget_subs, dim=-1)
    # distances.shape:  torch.Size([80, 80])

    mean_distances = distances.mean(dim=-1)

    i = torch.argmin(mean_distances)

    p_target_sub_reordered = reordered_ptarget_subs[i]
    # p_target_sub_reordered.shape:  torch.Size([80, 2])

    losses = torch.norm(p_pred - p_target_sub_reordered, dim=-1)

    return losses.mean()


# ----------------------------------------
def gaussian_pyramid_loss(recons, input):
    recons_clone = recons.clone()
    input_clone = input.clone()

    recon_loss = F.mse_loss(recons_clone, input_clone,
                            reduction='none').mean(dim=[1, 2, 3])

    for j in range(2, 5):
        recons_clone = dsample(recons_clone)
        input_clone = dsample(input_clone)

        recon_loss += F.mse_loss(recons_clone, input_clone,
                                 reduction='none').mean(dim=[1, 2, 3]) / j

    return recon_loss
