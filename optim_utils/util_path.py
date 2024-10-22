import numpy as np
import cv2
import torch

# import sys
# sys.path.append("..")


def get_cubic_segments_from_points(points):
    total_points = points.shape[0]
    seg_num = total_points // 3

    cubics = points.view(seg_num, 3, 2)
    next_points = torch.roll(points, -3, dims=0)[:seg_num*3:3]
    cubics = torch.cat([cubics, next_points.unsqueeze(1)], dim=1)

    return cubics.view(-1, 4, 2)


def sample_bezier(cubics, k=5):
    """
    Sample points on cubic Bezier curves.
    :param cubics: torch.Tensor, shape [num_curves, 4, 2], representing cubic Bezier curves.
    :param k: int, number of sample points per curve.
    :return: torch.Tensor, shape [num_curves * k, 2], representing the sampled points on the Bezier curves.
    """
    # shape [1, k, 1]
    ts = torch.linspace(0, 1, k).view(1, k, 1).to(cubics.device)

    P0, P1, P2, P3 = cubics[:, 0], cubics[:, 1], cubics[:, 2], cubics[:, 3]

    # Calculate cubic Bezier for all curves and all t values at once
    point = (1-ts)**3 * P0.unsqueeze(1) + 3*(1-ts)**2*ts * P1.unsqueeze(1) + \
        3*(1-ts)*ts**2 * P2.unsqueeze(1) + ts**3 * P3.unsqueeze(1)

    # Reshape the tensor to get points in [num_curves * k, 2] format
    point = point.reshape(-1, 2)

    # shape [num_curves * k, 2]
    return point


def is_clockwise(p):
    start, end = p[:-1], p[1:]
    return torch.stack([start, end], dim=-1).det().sum() > 0


def make_clockwise(p):
    if not is_clockwise(p):
        return p.flip(dims=[0])
    return p


def sample_points_by_length_distribution(p_target, n, device="cuda"):
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
    distances = torch.cdist(distr_pred.unsqueeze(-1),
                            distr_target.unsqueeze(-1))
    matching = distances.argmin(dim=-1)

    # Select subset of target points based on matching
    p_target_sub = p_target_clockwise[matching]

    return p_target_sub, matching


def get_length_distribution(p, normalize=True):
    start, end = p[:-1], p[1:]
    length_distr = torch.norm(end - start, dim=-1).cumsum(dim=0)
    length_distr = torch.cat([length_distr.new_zeros(1),
                              length_distr])

    if normalize:
        length_distr = length_distr / length_distr[-1]

    return length_distr


def fill_holes(thresh_image):
    thresh_filled = thresh_image.copy()
    h, w = thresh_image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(thresh_filled, mask, (0, 0), 255)
    thresh_filled_inv = cv2.bitwise_not(thresh_filled)
    thresh_image = thresh_image | thresh_filled_inv
    return thresh_image
