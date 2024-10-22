import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsvg.difflib.utils import make_clockwise, reorder, get_length_distribution


def vae_loss(recon_x, x, mu, logvar, lengths):
    mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand(-1, -1, x.size(2)).float().to(x.device)

    # Only consider non-padded elements for MSE loss
    recon_x_masked = recon_x * mask
    x_masked = x * mask
    BCE = F.mse_loss(recon_x_masked, x_masked, reduction='none')
    BCE = BCE.sum() / mask.sum()  # Mean over non-padded elements

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / x.size(0)  # Mean over batch

    return BCE + KLD


def vae_loss_nomask(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='none')
    BCE = BCE.sum(-1).mean()  # Mean over sequence length and batch
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / x.size(0)  # Normalize over batch
    return BCE + KLD


def vae_loss_nomask_sum(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def vae_loss_nomask_wt(recon_x, x, mu, log_var, kld_weight):

    recons_loss = F.mse_loss(recon_x, x)
    assert not torch.isnan(mu).any()
    assert not torch.isnan(log_var).any()

    mu2 = mu ** 2
    assert not torch.isnan(mu2).any()

    log_var_exp = log_var.exp()
    assert not torch.isnan(log_var_exp).any()

    term1 = 1 + log_var - mu2 - log_var_exp
    assert not torch.isnan(term1).any()

    kld_loss = torch.mean(-0.5 * torch.sum(term1, dim=1), dim=0)
    assert not torch.isnan(kld_loss).any()

    loss = recons_loss + kld_weight * kld_loss
    return loss


def mkld_loss(mu, log_var, kld_weight=0.00001, kl_tolerance=0.1, use_sum=True):
    epsilon = 1e-8

    kld_term = 1.0 + log_var - mu.pow(2) - torch.exp(log_var + epsilon)
    if (use_sum):
        # 每个batch内求和, 在批次维度上求平均
        kld_bat_sum = -0.5 * torch.sum(kld_term)
        kld_loss = kld_bat_sum / mu.size(0)
    else:
        # 所有样本的平均值
        kld_loss = -0.5 * torch.mean(kld_term)

    # kld_loss = torch.clamp(kld_loss, min=kl_tolerance)
    w_kld_loss = kld_weight * kld_loss

    return w_kld_loss


def vae_loss_mask_wt_transformer(recon_x, x, mu, log_var, lengths, kld_weight=0.00001, kl_tolerance=0.1, recons_loss_weight=1.0, use_sum=True):
    recons_loss = recons_loss_mask(recon_x, x, lengths, use_sum=use_sum)
    w_kld_loss = mkld_loss(mu=mu, log_var=log_var, kld_weight=kld_weight,
                           kl_tolerance=kl_tolerance, use_sum=use_sum)

    loss = recons_loss * recons_loss_weight + w_kld_loss

    return loss, recons_loss, w_kld_loss


def recons_loss_mask(recon_x, x, lengths, use_sum=True):
    mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand(-1, -1, x.size(2)).float().to(x.device)

    # Only consider non-padded elements for MSE loss
    recon_x_masked = recon_x * mask
    x_masked = x * mask

    if (use_sum):
        recons_loss = F.mse_loss(recon_x_masked, x_masked, reduction='sum')
        recons_loss = recons_loss / recon_x.size(0)
    else:
        recons_loss = F.mse_loss(recon_x_masked, x_masked, reduction='mean')

    return recons_loss


def vae_loss_mask_wt_transformer_weight(recon_x, x, mu, log_var, lengths, kld_weight=0.00001, kl_tolerance=0.1, recons_loss_weight=1.0, z_weight=1.0, color_weight=1.0, affine_weight=1.0, use_sum=True):
    recons_loss = recons_loss_mask_weight(recon_x=recon_x, x=x, lengths=lengths, z_weight=z_weight,
                                          color_weight=color_weight, affine_weight=affine_weight, use_sum=use_sum)

    w_kld_loss = mkld_loss(mu=mu, log_var=log_var, kld_weight=kld_weight,
                           kl_tolerance=kl_tolerance, use_sum=use_sum)

    loss = recons_loss_weight * recons_loss + w_kld_loss

    return loss, recons_loss, w_kld_loss


def recons_loss_mask_weight(recon_x, x, lengths, z_weight=1.0, color_weight=1.0, affine_weight=1.0, use_sum=True):
    mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand(-1, -1, x.size(2)).float().to(x.device)

    # Only consider non-padded elements for MSE loss
    recon_x_masked = recon_x * mask
    x_masked = x * mask
    # x_masked.shape:  torch.Size([512, 25, 32])

    recon_x_masked_z = recon_x_masked[:, :, :-8]
    x_masked_z = x_masked[:, :, :-8]
    recon_x_masked_color = recon_x_masked[:, :, -8:-4]
    x_masked_color = x_masked[:, :, -8:-4]
    recon_x_masked_affine = recon_x_masked[:, :, -4:]
    x_masked_affine = x_masked[:, :, -4:]

    if (use_sum):
        # recons_loss = F.mse_loss(recon_x_masked, x_masked, reduction='sum')
        recons_loss_z = F.mse_loss(
            recon_x_masked_z, x_masked_z, reduction='sum')
        recons_loss_color = F.mse_loss(
            recon_x_masked_color, x_masked_color, reduction='sum')
        recons_loss_affine = F.mse_loss(
            recon_x_masked_affine, x_masked_affine, reduction='sum')
        recons_loss_z = recons_loss_z / recon_x.size(0)
        recons_loss_color = recons_loss_color / recon_x.size(0)
        recons_loss_affine = recons_loss_affine / recon_x.size(0)

    else:
        # recons_loss = F.mse_loss(recon_x_masked, x_masked, reduction='mean')
        recons_loss_z = F.mse_loss(
            recon_x_masked_z, x_masked_z, reduction='mean')
        recons_loss_color = F.mse_loss(
            recon_x_masked_color, x_masked_color, reduction='mean')
        recons_loss_affine = F.mse_loss(
            recon_x_masked_affine, x_masked_affine, reduction='mean')

    recons_loss = z_weight * recons_loss_z + color_weight * \
        recons_loss_color + affine_weight * recons_loss_affine

    return recons_loss


# ----------------------------------------
def is_clockwise_batch(p):
    """
    Determine if each polygon in a batch is clockwise.

    :param p: torch.Tensor
        A batch of polygons, each represented by a series of points.
        Shape: [batch_size, num_points, 2]
    :return: torch.Tensor
        A tensor of shape [batch_size] containing boolean values.
    """
    start, end = p[:, :-1], p[:, 1:]
    det_values = torch.stack([start, end], dim=-2).det()
    return det_values.sum(dim=-1) > 0


def make_clockwise_batch(p):
    """
    Ensure the polygons are clockwise.
    :param p: torch.Tensor, shape [batch_size, num_points, 2] representing polygons.
    :return: torch.Tensor, shape [batch_size, num_points, 2], clockwise polygons.
    """
    clockwise_mask = is_clockwise_batch(p)
    reversed_p = p.flip(dims=[1])
    # Using view to reshape the mask
    return torch.where(clockwise_mask.unsqueeze(-1).unsqueeze(-1), p, reversed_p)


def get_length_batch(p):
    start, end = p[:, :-1], p[:, 1:]
    return torch.norm(end - start, dim=-1).sum(dim=1)


def reorder_batch(p, i):
    b, n, _ = p.size()

    # Ensure i is a tensor of shape (b,)
    if not isinstance(i, torch.Tensor):
        i = torch.full((b,), i, dtype=torch.long, device=p.device)

    # Expand the indices to shape (b, n)
    expanded_i = i.unsqueeze(-1).repeat(1, n)

    # Create a range tensor of shape (b, n)
    range_tensor = torch.arange(n, device=p.device).unsqueeze(0).repeat(b, 1)

    # Compute the reordered indices consistent with the 'reorder' function
    reordered_indices = (range_tensor - expanded_i) % n

    # Use gather to reorder the batch of tensors
    reordered_p = torch.gather(
        p, 1, reordered_indices.unsqueeze(-1).expand(-1, -1, p.size(2)))

    return reordered_p


def get_length_distribution_batch(p, normalize=True):
    """
    Get length distribution for a batch of points.
    """
    start, end = p[:, :-1], p[:, 1:]
    length_distr = torch.norm(end - start, dim=-1).cumsum(dim=1)
    length_distr = torch.cat([length_distr.new_zeros((p.shape[0], 1)),
                              length_distr], dim=1)

    if normalize:
        length_distr = length_distr / length_distr[:, -1].unsqueeze(1)

    return length_distr


def get_target_distr(p_target, n, device="cuda"):
    """
    Compute a subset of target points based on length distribution.

    Args:
        p_target (torch.Tensor): Target points, shape [num_points, 2].
        n (int): Number of points to sample.
        device (str): Device to use for computations.

    Returns:
        tuple: (p_target_sub, matching)
            p_target_sub (torch.Tensor): Subset of target points, shape [n, 2].
            matching (torch.Tensor): Indices of matched points, shape [n].
    """
    assert n > 0, "n must be positive"

    # Assume p_target is already clockwise
    p_target_clockwise = p_target

    # Create evenly spaced distribution for predicted points
    distr_pred = torch.linspace(0., 1., n, device=device)

    # Compute cumulative length distribution of target points
    distr_target = get_length_distribution(p_target_clockwise, normalize=True)

    # Find closest target point for each predicted point
    distances = torch.cdist(distr_pred.unsqueeze(-1), distr_target.unsqueeze(-1))
    matching = distances.argmin(dim=-1)

    # Select subset of target points based on matching
    p_target_sub = p_target_clockwise[matching]

    return p_target_sub, matching


def get_target_distr_batch(p_target, n, device="cuda"):
    """
    Get target distribution for a batch of points.

    Parameters:
    - p_target: A tensor of shape [batch_size, num_points, 2], representing a batch of point sets.
    - n: The number of divisions for linspace.
    - device: The device to which tensors should be moved.

    Returns:
    - p_target_sub: A tensor of shape [batch_size, n, 2], representing the subset of target points.
    - matching: A tensor of shape [batch_size, n], representing the index of the closest point in p_target to each point in distr_pred.
    """
    b = p_target.size(0)
    assert n > 0, "n must be positive"

    # Make target point lists clockwise
    # p_target_clockwise = make_clockwise_batch(p_target)

    p_target_clockwise = p_target
    # assert is_clockwise_batch(p_target_clockwise)

    # Compute length distribution
    distr_pred = torch.linspace(0., 1., n).to(
        device).unsqueeze(0).expand(b, -1)
    distr_target = get_length_distribution_batch(
        p_target_clockwise, normalize=True)

    # Compute matching
    d = torch.cdist(distr_pred.unsqueeze(-1), distr_target.unsqueeze(-1))
    matching = d.argmin(dim=-1)

    # Gather p_target_sub values based on matching indices
    p_target_sub = torch.gather(
        p_target_clockwise, 1, matching.unsqueeze(-1).expand(b, n, 2))

    return p_target_sub, matching


def svg_emd_loss_v1(p_pred, p_target, p_target_sub=None, matching=None,
                    first_point_weight=False, return_matched_indices=False):

    n, m = len(p_pred), len(p_target)

    if n == 0:
        return 0.

    if p_target_sub is None or matching is None:
        p_target_sub, matching = get_target_distr(
            p_target, n, device=p_pred.device)

    # EMD
    # Find the best reorder (保持p_pred点的顺序?)
    i = torch.argmin(torch.stack(
        [torch.norm(p_pred - reorder(p_target_sub, i), dim=-1).mean() for i in range(n)]))

    p_target_sub_reordered = reorder(p_target_sub, i)
    losses = torch.norm(p_pred - p_target_sub_reordered, dim=-1)

    if first_point_weight:
        weights = torch.ones_like(losses)
        weights[0] = 10.
        losses = losses * weights

    if return_matched_indices:
        return losses.mean(), (p_pred, p_target, reorder(matching, i))

    return losses.mean()


def svg_emd_loss(p_pred, p_target, p_target_sub=None, matching=None):

    n, m = len(p_pred), len(p_target)

    if n == 0:
        return 0.

    if p_target_sub is None or matching is None:
        p_target_sub, matching = get_target_distr(
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


def svg_emd_loss_batch_v1(p_pred, p_target, p_target_sub=None, matching=None,
                          first_point_weight=False, return_matched_indices=False):
    b, n, _ = p_pred.size()

    if n == 0:
        return 0.

    if p_target_sub is None or matching is None:
        p_target_sub, matching = get_target_distr_batch(
            p_target, n, device=p_pred.device)

    # EMD
    # Compute reorder loss for every possible reordering
    reorder_loss = torch.stack(
        [torch.norm(p_pred - reorder_batch(p_target_sub, i),
                    dim=-1).mean(dim=1) for i in range(n)],
        dim=1
    )
    best_reorder_indices = torch.argmin(reorder_loss, dim=1)

    # Reorder based on the best indices
    p_target_sub_reordered = reorder_batch(p_target_sub, best_reorder_indices)

    losses = torch.norm(p_pred - p_target_sub_reordered, dim=-1)

    if first_point_weight:
        weights = torch.ones_like(losses)
        weights[:, 0] = 10.
        losses *= weights

    if return_matched_indices:
        reordered_matching = reorder_batch(matching, best_reorder_indices)
        return losses.mean(), (p_pred, p_target, reordered_matching)

    return losses.mean()


def svg_emd_loss_batch(p_pred, p_target, p_target_sub=None, matching=None):
    """
    Compute the EMD loss for a batch of predictions and targets using vectorized reordering.

    Parameters:
    - p_pred (torch.Tensor): A batch of predicted point sets of shape [batch_size, num_points, 2].
    - p_target (torch.Tensor): A batch of target point sets of shape [batch_size, num_points, 2].
    - p_target_sub (Optional[torch.Tensor]): A batch of subset target points. Shape: [batch_size, num_points, 2].
    - matching (Optional[torch.Tensor]): Indices of the matchings between predictions and targets. Shape: [batch_size, num_points].

    Returns:
    - torch.Tensor: A scalar tensor representing the mean EMD loss across the batch.
    """
    b, n, _ = p_pred.shape

    if n == 0:
        return torch.tensor(0., device=p_pred.device)

    if p_target_sub is None or matching is None:
        p_target_sub, matching = get_target_distr_batch(
            p_target, n, device=p_pred.device)

    # Vectorized computation to get reorderings for each set in the batch
    indices = torch.arange(n, device=p_pred.device).unsqueeze(0).repeat(n, 1)
    roll_indices = (indices + indices.T) % n
    roll_indices_batch = roll_indices.unsqueeze(0).repeat(b, 1, 1)

    reordered_ptarget_subs = p_target_sub[torch.arange(
        b)[:, None, None], roll_indices_batch]

    # print("roll_indices_batch.shape: ", roll_indices_batch.shape)
    # print("reordered_ptarget_subs.shape: ", reordered_ptarget_subs.shape)
    # roll_indices_batch.shape:  torch.Size([12, 80, 80])
    # reordered_ptarget_subs.shape:  torch.Size([12, 80, 80, 2])

    # Compute distances for all reorderings and find the optimal permutation
    distances = torch.norm(
        p_pred[:, None, :, :] - reordered_ptarget_subs, dim=-1)

    # print("distances.shape: ", distances.shape)
    # distances.shape:  torch.Size([12, 80, 80])

    mean_distances = distances.mean(dim=-1)
    # Get the index of the minimum mean distance for each batch
    i = torch.argmin(mean_distances, dim=-1)

    # Use advanced indexing to select the optimal reordered points for each sample
    batch_indices = torch.arange(b, device=p_pred.device)
    p_target_sub_reordered = reordered_ptarget_subs[batch_indices, :, i]

    # print("p_target_sub_reordered.shape: ", p_target_sub_reordered.shape)
    # p_target_sub_reordered.shape:  torch.Size([12, 80, 2])

    # print("p_pred.shape: ", p_pred.shape)
    # p_pred.shape:  torch.Size([12, 80, 2])

    losses_batch = torch.norm(
        p_pred - p_target_sub_reordered, dim=-1).mean(dim=-1)
    # print("losses_batch.shape: ", losses_batch.shape)
    # losses_batch.shape:  torch.Size([12])

    return losses_batch.mean()


def simplified_svg_emd_loss(p_pred, p_target, p_target_sub=None, matching=None):
    n, m = len(p_pred), len(p_target)

    if n == 0:
        return 0.

    if p_target_sub is None or matching is None:
        p_target_sub, matching = get_target_distr(
            p_target, n, device=p_pred.device)

    # 找到p_target_sub中距离p_pred的第一个点最近的那个点的索引
    distances_to_first_point = torch.norm(p_target_sub - p_pred[0], dim=-1)
    start_idx = torch.argmin(distances_to_first_point)

    # 使用找到的起始点重新排序p_target_sub
    reordered_p_target_sub = torch.roll(
        p_target_sub, shifts=(-start_idx.item(),), dims=0)

    # 计算两个路径之间的点对点距离
    losses = torch.norm(p_pred - reordered_p_target_sub, dim=-1)

    return losses.mean()


def svg_emd_loss_same_v1(p_pred, p_target, first_point_weight=False, return_matched_indices=False):
    # p_pred 和 p_target 有相同数量的点
    n, _ = p_pred.size()

    if n == 0:
        return 0.

    p_target_clockwise = make_clockwise(p_target)

    # Compute reorder loss for every possible reordering
    all_losses = torch.stack(
        [torch.norm(p_pred - reorder(p_target_clockwise, i), dim=-1).mean()
         for i in range(n)]
    )

    best_reorder_index = torch.argmin(all_losses)
    p_target_reordered = reorder(p_target_clockwise, best_reorder_index)

    losses = torch.norm(p_pred - p_target_reordered, dim=-1)

    if first_point_weight:
        weights = torch.ones_like(losses)
        weights[0] = 10.
        losses *= weights

    if return_matched_indices:
        reordered_matching = reorder(torch.arange(
            n, device=p_pred.device), best_reorder_index)
        return losses.mean(), (p_pred, p_target, reordered_matching)

    return losses.mean()


def svg_emd_loss_same(p_pred, p_target):
    # p_pred 和 p_target 有相同数量的点
    n, _ = p_pred.size()

    if n == 0:
        return 0.

    # p_target_clockwise = make_clockwise(p_target)
    p_target_clockwise = p_target
    # assert is_clockwise(p_target_clockwise)

    # Create a tensor containing all reordered versions of p_target_clockwise
    indices = torch.arange(n, device=p_pred.device).unsqueeze(0).repeat(n, 1)
    roll_indices = (indices + indices.T) % n
    reordered_ptargets = torch.index_select(
        p_target_clockwise, 0, roll_indices.view(-1)).view(n, n, 2)

    # Compute distances for all reorderings
    distances = torch.norm(p_pred.unsqueeze(0) - reordered_ptargets, dim=-1)
    all_losses = distances.mean(dim=-1)

    best_reorder_index = torch.argmin(all_losses)
    p_target_reordered = reordered_ptargets[best_reorder_index]

    losses = torch.norm(p_pred - p_target_reordered, dim=-1)

    return losses.mean()


def continuity_loss(cubics):
    """Compute the loss for ensuring continuity between cubic segments using vectorized operations."""

    # Extract the end points of all segments except the last
    end_points = cubics[:-1, 3]

    # Extract the start points of all segments except the first
    start_points_of_next_segments = cubics[1:, 0]

    # Compute the L2 distance for all segments simultaneously
    loss = torch.norm(
        end_points - start_points_of_next_segments, p=2, dim=1).sum()

    return loss


def continuity_loss_batch(cubics_batch):
    """Compute the loss for ensuring continuity between cubic segments on batch level using vectorized operations.

    :param cubics_batch: torch.Tensor, shape [batch_size, num_segments, 4, point_dim]
    :return: torch.Tensor, shape [batch_size], continuity loss for each batch
    """
    # Extract the end points of all segments except the last for each batch
    end_points = cubics_batch[:, :-1, 3]

    # Extract the start points of all segments except the first for each batch
    start_points_of_next_segments = cubics_batch[:, 1:, 0]

    # Compute the L2 distance for all segments simultaneously for each batch
    loss_batch = torch.norm(
        end_points - start_points_of_next_segments, p=2, dim=-1).sum(dim=-1)

    return loss_batch.mean()
