import copy
import shutil
import numpy.random as npr
import numpy as np
import os.path as osp
import os
import PIL.Image
import PIL
import pydiffvg
import torch
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torchvision import transforms


import warnings
warnings.filterwarnings("ignore")


pydiffvg.set_print_timing(False)
gamma = 1.0


def get_experiment_id(debug=False):
    if debug:
        return 999999999999
    import time
    time.sleep(0.5)
    return int(time.time()*100)


def get_path_schedule(type, **kwargs):
    if type == 'repeat':
        max_path = kwargs['max_path']
        schedule_each = kwargs['schedule_each']
        return [schedule_each] * max_path
    elif type == 'list':
        schedule = kwargs['schedule']
        return schedule
    elif type == 'exp':
        import math
        base = kwargs['base']
        max_path = kwargs['max_path']
        max_path_per_iter = kwargs['max_path_per_iter']
        schedule = []
        cnt = 0
        while sum(schedule) < max_path:
            proposed_step = min(
                max_path - sum(schedule),
                base**cnt,
                max_path_per_iter)
            cnt += 1
            schedule += [proposed_step]
        return schedule
    else:
        raise ValueError


def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append(edict_2_dict(x[i]))
        return xnew
    else:
        return x


def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)


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


class random_coord_init():
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size

    def __call__(self):
        h, w = self.canvas_size
        return [npr.uniform(0, 1) * w, npr.uniform(0, 1) * h]


class naive_coord_init():
    def __init__(self, pred, gt, format='[bs x c x 2D]',
                 replace_sampling=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0])**2).sum(0)
        elif format == ['[2D x c]']:
            self.map = ((pred - gt)**2).sum(-1)
        else:
            raise ValueError
        self.replace_sampling = replace_sampling

    def __call__(self):
        coord = np.where(self.map == self.map.max())
        coord_h, coord_w = coord[0][0], coord[1][0]
        if self.replace_sampling:
            self.map[coord_h, coord_w] = -1
        return [coord_w, coord_h]


class sparse_coord_init():
    def __init__(self,
                 pred,
                 gt,
                 format='[bs x c x 2D]',
                 quantile_interval=200,
                 nodiff_thres=0.1):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0])**2).sum(0)
            self.reference_gt = copy.deepcopy(np.transpose(gt[0], (1, 2, 0)))
        elif format == ['[2D x c]']:
            self.map = (np.abs(pred - gt)).sum(-1)
            self.reference_gt = copy.deepcopy(gt[0])
        else:
            raise ValueError
        # OptionA: Zero too small errors to avoid the error too small deadloop
        self.map[self.map < nodiff_thres] = 0
        quantile_interval = np.linspace(0., 1., quantile_interval)
        quantized_interval = np.quantile(self.map, quantile_interval)
        # remove redundant
        quantized_interval = np.unique(quantized_interval)
        quantized_interval = sorted(quantized_interval[1:-1])
        self.map = np.digitize(self.map, quantized_interval, right=False)
        self.map = np.clip(self.map, 0, 255).astype(np.uint8)
        self.idcnt = {}
        for idi in sorted(np.unique(self.map)):
            self.idcnt[idi] = (self.map == idi).sum()
        self.idcnt.pop(min(self.idcnt.keys()))
        # remove smallest one to remove the correct region

    def __call__(self):
        if len(self.idcnt) == 0:
            h, w = self.map.shape
            return [npr.uniform(0, 1) * w, npr.uniform(0, 1) * h]
        target_id = max(self.idcnt, key=self.idcnt.get)
        _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
            (self.map == target_id).astype(np.uint8), connectivity=4)
        # remove cid = 0, it is the invalid area
        csize = [ci[-1] for ci in cstats[1:]]
        target_cid = csize.index(max(csize)) + 1
        center = ccenter[target_cid][::-1]
        coord = np.stack(np.where(component == target_cid)).T
        dist = np.linalg.norm(coord - center, axis=1)
        target_coord_id = np.argmin(dist)
        coord_h, coord_w = coord[target_coord_id]
        # replace_sampling
        self.idcnt[target_id] -= max(csize)
        if self.idcnt[target_id] == 0:
            self.idcnt.pop(target_id)
        self.map[component == target_cid] = 0
        return [coord_w, coord_h]


def load_target_new(fp, img_size=224):
    target = PIL.Image.open(fp).resize((img_size, img_size))
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    transforms_ = [transforms.ToTensor()]
    data_transforms = transforms.Compose(transforms_)  # w,h,c -> c,h,w
    target = data_transforms(target)
    target = target.unsqueeze(0)
    return target


def init_shapes(num_paths,
                num_segments,
                canvas_size,
                seginit_cfg,
                shape_cnt,
                pos_init_method=None,
                trainable_stroke=False,
                gt=None,
                **kwargs):

    seginit_cfg = type('', (), seginit_cfg)()

    shapes = []
    shape_groups = []
    h, w = canvas_size

    # change path init location
    if pos_init_method is None:
        pos_init_method = random_coord_init(canvas_size=canvas_size)

    for i in range(num_paths):
        num_control_points = [2] * num_segments

        if seginit_cfg.type == "random":
            points = []
            p0 = pos_init_method()
            color_ref = copy.deepcopy(p0)
            points.append(p0)
            for j in range(num_segments):
                radius = seginit_cfg.radius
                p1 = (p0[0] + radius * npr.uniform(-0.5, 0.5),
                      p0[1] + radius * npr.uniform(-0.5, 0.5))
                p2 = (p1[0] + radius * npr.uniform(-0.5, 0.5),
                      p1[1] + radius * npr.uniform(-0.5, 0.5))
                p3 = (p2[0] + radius * npr.uniform(-0.5, 0.5),
                      p2[1] + radius * npr.uniform(-0.5, 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.FloatTensor(points)

        # circle points initialization
        elif seginit_cfg.type == "circle":
            radius = seginit_cfg.radius
            if radius is None:
                radius = npr.uniform(0.5, 1)
            center = pos_init_method()
            # print("center = ", center)
            # center =  [20.810912607961424, 41.24654946404007]
            color_ref = copy.deepcopy(center)
            points = get_bezier_circle(radius=radius,
                                       segments=num_segments,
                                       bias=center)

        path = pydiffvg.Path(
            num_control_points=torch.LongTensor(num_control_points),
            points=points,
            stroke_width=torch.tensor(0.0),
            is_closed=True)
        shapes.append(path)

        # if 'gt' in kwargs:
        if gt is not None:
            wref, href = color_ref
            wref = max(0, min(int(wref), w - 1))
            href = max(0, min(int(href), h - 1))
            fill_color_init = list(gt[0, :, href, wref]) + [1.]
            fill_color_init = torch.FloatTensor(fill_color_init)
            stroke_color_init = torch.FloatTensor(npr.uniform(size=[4]))
        else:
            fill_color_init = torch.FloatTensor(npr.uniform(size=[4]))
            stroke_color_init = torch.FloatTensor(npr.uniform(size=[4]))

        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([shape_cnt + i]),
            fill_color=fill_color_init,
            stroke_color=stroke_color_init,
        )
        shape_groups.append(path_group)

    point_var = []
    color_var = []

    for path in shapes:
        path.points.requires_grad = True
        point_var.append(path.points)
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_var.append(group.fill_color)

    if trainable_stroke:
        stroke_width_var = []
        stroke_color_var = []
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_var.append(path.stroke_width)
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            stroke_color_var.append(group.stroke_color)
        return shapes, shape_groups, point_var, color_var, stroke_width_var, stroke_color_var
    else:
        return shapes, shape_groups, point_var, color_var


# ----------------------------------------------------------

class linear_decay_lrlambda_f(object):
    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n // self.decay_every
        decay_step = n % self.decay_every
        lr_s = self.decay_ratio**decay_time
        lr_e = self.decay_ratio**(decay_time + 1)
        r = decay_step / self.decay_every
        lr = lr_s * (1 - r) + lr_e * r
        return lr

# ----------------------------------------------------------
