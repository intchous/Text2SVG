from sklearn.preprocessing import MinMaxScaler
import os
import random
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import PIL
import PIL.Image

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

import pydiffvg

pydiffvg.set_print_timing(False)
gamma = 1.0


class Normalize(object):
    def __init__(self, w, h):
        self.w = w * 1.0
        self.h = h * 1.0
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def __call__(self, points):
        points = points / \
            torch.tensor([self.w, self.h], dtype=torch.float32).to(
                points.device)

        return points

    def inverse_transform(self, points):

        points = points * \
            (torch.tensor([self.w, self.h],
             dtype=torch.float32).to(points.device))

        return points


def load_target_new(fp, img_size=64):
    target = PIL.Image.open(fp).resize((img_size, img_size))
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # target = np.array(target)

    transforms_ = []
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)  # w,h,c -> c,h,w
    target = data_transforms(target)
    target = target[0]
    target = target.unsqueeze(0)
    # gt = data_transforms(target).unsqueeze(0).to(device)
    # print("gt.shape = ", gt.shape)
    return target


def load_tartget(fp):
    target = PIL.Image.open(fp)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # target = np.array(target)

    transforms_ = []
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)  # w,h,c -> c,h,w
    target = data_transforms(target)
    return target


class SVGDataset_nopadding(Dataset):
    def __init__(self, directory, h, w, fixed_length=60, file_list=None, img_dir=None, transform=None, use_model_fusion=False):
        super(SVGDataset_nopadding, self).__init__()
        self.directory = directory

        if file_list is None:
            self.file_list = os.listdir(self.directory)
        else:
            self.file_list = file_list

        self.transform = transform
        self.h = h
        self.w = w
        self.fixed_length = fixed_length
        self.img_dir = img_dir
        self.use_model_fusion = use_model_fusion

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.directory, self.file_list[idx])

        try:
            assert os.path.exists(filepath)

            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
                filepath)

            for path in shapes:
                points = path.points
                num_control_points = path.num_control_points
                break

            # Transform points if applicable
            if self.transform:
                points = self.transform(points)

            # fixed_length 是 控制点的个数
            # Truncate if sequence is too long
            if points.shape[0] > self.fixed_length:
                points = points[:self.fixed_length]

            # assert points.shape[0] == self.fixed_length

            # Compute the cubics segments
            cubics = get_cubic_segments_from_points(
                points=points, num_control_points=num_control_points)
            # cubics.shape: torch.Size([10, 4, 2])

            desired_cubics_length = self.fixed_length // 3

            assert cubics.shape[0] == desired_cubics_length

            path_img = []
            if self.img_dir:
                im_pre = self.file_list[idx].split(".")[0]
                im_path = os.path.join(self.img_dir, im_pre + ".png")
                if (os.path.exists(im_path)):
                    if (self.use_model_fusion):
                        path_img = load_target_new(im_path)
                    else:
                        path_img = load_tartget(im_path)

            res_data = {
                # control points
                "points": points,
                # cubics segments
                "cubics": cubics,
                # 原始长度 (控制点)
                "lengths": self.fixed_length,
                "filepaths": filepath,
                "path_img": path_img
            }

        except Exception as e:
            print(f"Error processing index: {idx}, Filepath: {filepath}")
            print(f"Error message: {str(e)}")
            raise e

        return res_data


class SVGDataset(Dataset):
    def __init__(self, directory, h, w, fixed_length=60, file_list=None, img_dir=None, transform=None, use_model_fusion=False):
        super(SVGDataset, self).__init__()
        self.directory = directory

        if file_list is None:
            self.file_list = os.listdir(self.directory)
        else:
            self.file_list = file_list

        self.transform = transform
        self.h = h
        self.w = w
        self.fixed_length = fixed_length
        self.img_dir = img_dir
        self.use_model_fusion = use_model_fusion

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.directory, self.file_list[idx])

        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
            filepath)

        for path in shapes:
            points = path.points
            num_control_points = path.num_control_points
            break

        # Transform points if applicable
        if self.transform:
            points = self.transform(points)

        # fixed_length 是 控制点的个数
        # Truncate if sequence is too long
        len_points = points.shape[0]
        if len_points > self.fixed_length - 1:
            points = points[:self.fixed_length - 1]

        # Compute the cubics segments
        cubics = get_cubic_segments_from_points(
            points=points, num_control_points=num_control_points)

        # Determine the desired number of cubics based on fixed_length and existing points
        desired_cubics_length = (self.fixed_length - 1) // 3

        # Pad or truncate cubics based on desired length
        if cubics.shape[0] < desired_cubics_length:
            padding_needed = desired_cubics_length - cubics.shape[0]
            # Using zero-padding
            padding_tensor = torch.full((padding_needed, 4, 2), 0.0)
            cubics = torch.cat([cubics, padding_tensor], dim=0)
        elif cubics.shape[0] > desired_cubics_length:
            cubics = cubics[:desired_cubics_length]

        # Append end token (0.0, 0.0)
        points = torch.cat((points, torch.tensor(
            [[0.0, 0.0]], dtype=torch.float32)))

        len_points = points.shape[0]
        # Pad to fixed_length
        if (len_points < self.fixed_length):
            points = torch.cat((points, torch.tensor(
                [[0.0, 0.0]] * (self.fixed_length - len_points), dtype=torch.float32)))

        assert points.shape[0] == self.fixed_length

        path_img = []
        if self.img_dir:
            im_pre = self.file_list[idx].split(".")[0]
            im_path = os.path.join(self.img_dir, im_pre + ".png")
            if (os.path.exists(im_path)):
                if (self.use_model_fusion):
                    path_img = load_target_new(im_path)
                else:
                    path_img = load_tartget(im_path)

        res_data = {
            # control points
            "points": points,
            # cubics segments
            "cubics": cubics,
            # 原始长度 + 1
            "lengths": len_points,
            "filepaths": filepath,
            "path_img": path_img
        }

        return res_data


def collate_fn_rnnpad(batch):
    # Append end token to each sequence in the batch
    batch = [torch.cat((seq, torch.tensor(
        [[-1., -1.]], dtype=torch.float32)), dim=0) for seq in batch]

    # Compute sequence lengths
    lengths = [len(seq) for seq in batch]

    # Pad the sequences
    batch = nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=-1)

    return batch, lengths


def collate_fn(batch):
    # Fixed length to which sequences should be padded
    fixed_length = 30

    # Truncate too long sequences, append end token and pad each sequence in the batch to fixed_length
    padded_batch = []
    for seq in batch:
        # Truncate if sequence is too long
        if len(seq) > fixed_length - 1:
            seq = seq[:fixed_length - 1]

        # Append end token
        seq = torch.cat((seq, torch.tensor([[0.0, 0.0]], dtype=torch.float32)))

        # Pad to fixed_length
        seq = torch.cat((seq, torch.tensor(
            [[0.0, 0.0]] * (fixed_length - len(seq)), dtype=torch.float32)))

        padded_batch.append(seq)

    # Convert list of tensors to a single tensor
    padded_batch = torch.stack(padded_batch)

    # Compute sequence lengths
    lengths = [len(seq) for seq in padded_batch]

    return padded_batch, lengths


# ----------------------------------------
def get_segments(pathObj):
    segments = []
    lines = []
    quadrics = []
    cubics = []
    # segList = (lines, quadrics, cubics)
    idx = 0
    total_points = pathObj.points.shape[0]
    # pathObj.points.shape:  torch.Size([21, 2])

    for ncp in pathObj.num_control_points.numpy():
        pt1 = pathObj.points[idx]

        if ncp == 0:
            segments.append((0, len(lines)))
            pt2 = pathObj.points[(idx + 1) % total_points]
            lines.append(pt1)
            lines.append(pt2)
            # lines.append((pt1, pt2))
            # lines.append((idx, (idx+1) % total_points))
            idx += 1
        elif ncp == 1:
            segments.append((1, len(quadrics)))
            pt2 = pathObj.points[idx + 1]
            pt3 = pathObj.points[(idx + 2) % total_points]
            quadrics.append(pt1)
            quadrics.append(pt2)
            quadrics.append(pt3)
            # quadrics.append((pt1, pt2, pt3))
            # quadrics.append((idx, (idx+1), (idx+2) % total_points))
            idx += ncp+1
        elif ncp == 2:
            segments.append((2, len(cubics)))
            pt2 = pathObj.points[idx + 1]
            pt3 = pathObj.points[idx + 2]
            pt4 = pathObj.points[(idx + 3) % total_points]
            cubics.append(pt1)
            cubics.append(pt2)
            cubics.append(pt3)
            cubics.append(pt4)

            # cubics.append((pt1, pt2, pt3, pt4))
            # cubics.append((idx, (idx+1), (idx+2), (idx+3) % total_points))
            idx += ncp + 1

    # total_points/3*4
    cubics = torch.stack(cubics).view(-1, 4, 2)
    return cubics


def get_cubic_segments_mask(lengths, max_pts_len_thresh, device="cuda"):
    cubic_lengths = (lengths - 1) // 3
    # cubic_lengths:  tensor([ 4, 10, 10,  6,  8,  6, 10,  6])
    max_cubics_length = (max_pts_len_thresh - 1) // 3
    # max_cubics_length:  20

    # Create the mask tensor
    cubics_mask = torch.arange(max_cubics_length).unsqueeze(
        0) < cubic_lengths.unsqueeze(1)

    # Expand dimensions to match cubics tensor shape
    cubics_mask = cubics_mask.unsqueeze(
        -1).unsqueeze(-1).expand(-1, -1, 4, 2).float()

    cubics_mask = cubics_mask.to(device)
    # print("cubics_mask.shape: ", cubics_mask.shape)
    # cubics_mask.shape:  torch.Size([8, 20, 4, 2])

    return cubics_mask, cubic_lengths


def get_cubic_segments_from_points(points, num_control_points):
    cubics = []
    idx = 0
    total_points = points.shape[0]
    # points.shape:  torch.Size([21, 2])

    for ncp in num_control_points.numpy():
        assert ncp == 2

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


def get_cubic_segments(pathObj):
    cubics = get_cubic_segments_from_points(
        points=pathObj.points, num_control_points=pathObj.num_control_points)

    return cubics


def cubic_segments_to_points(cubics):
    num_segments = cubics.shape[0]
    points_list = []

    # 处理第一个segment
    first_segment = cubics[0]
    points_list.extend([pt for pt in first_segment[:-1]])  # 只添加前三个点

    # 遍历其他所有的segment
    for idx in range(1, num_segments):
        prev_end = cubics[idx-1][3]  # 前一个段的ed点
        current_start = cubics[idx][0]  # 当前段的st点

        # 计算共用点
        shared_point = (prev_end + current_start) / 2.0
        points_list.append(shared_point)

        # 加入当前段的control1, control2
        points_list.extend([pt for pt in cubics[idx][1:3]])

    convert_points = torch.stack(points_list)

    return convert_points


def cubics_to_pathObj(cubics):
    """Given a cubics tensor, return a pathObj."""
    convert_points = cubic_segments_to_points(cubics)

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


def cubic_bezier(t, P0, P1, P2, P3):
    """
    Compute point on a cubic Bezier curve.
    :param t: torch.Tensor, Parameter t in range [0,1].
    :param P0: torch.Tensor, Start Point.
    :param P1: torch.Tensor, Control Point 1.
    :param P2: torch.Tensor, Control Point 2.
    :param P3: torch.Tensor, End Point.
    :return: torch.Tensor, the corresponding point on the cubic Bezier curve.
    """
    t_complement = 1 - t
    B = (
        t_complement ** 3 * P0
        + 3 * t_complement ** 2 * t * P1
        + 3 * t_complement * t ** 2 * P2
        + t ** 3 * P3
    )
    return B


def sample_bezier(cubics, k=5):
    """
    Sample points on cubic Bezier curves.
    :param cubics: torch.Tensor, shape [num_curves, 4, 2], representing cubic Bezier curves.
    :param k: int, number of sample points per curve.
    :return: torch.Tensor, shape [num_curves * k, 2], representing the sampled points on the Bezier curves.
    """
    num_curves = cubics.shape[0]
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


def sample_bezier_batch(cubics, k=5):
    """
    Sample points on cubic Bezier curves.
    :param cubics: torch.Tensor, shape [batch_size, num_curves, 4, 2], representing cubic Bezier curves.
    :param k: int, number of sample points per curve.
    :return: torch.Tensor, shape [batch_size, num_curves * k, 2], representing the sampled points on the Bezier curves.
    """
    batch_size, num_curves = cubics.shape[0], cubics.shape[1]

    # shape [1, 1, k, 1]
    ts = torch.linspace(0, 1, k).view(1, 1, k, 1).to(cubics.device)
    t_inv = 1 - ts

    # Break down the cubics tensor for cubic bezier formula
    P0, P1, P2, P3 = cubics[:, :, 0], cubics[:,
                                             :, 1], cubics[:, :, 2], cubics[:, :, 3]

    # Expand dimensions of P0, P1, P2, P3 for broadcasting with ts
    # shape [batch_size, num_curves, 1, 2]
    P0, P1, P2, P3 = P0[:, :, None, :], P1[:, :,
                                           None, :], P2[:, :, None, :], P3[:, :, None, :]

    # Using the cubic bezier formula
    sampled_points = t_inv**3 * P0 + 3 * t_inv**2 * \
        ts * P1 + 3 * t_inv * ts**2 * P2 + ts**3 * P3

    # Reshape to [batch_size, num_curves * k, 2]
    sampled_points = sampled_points.reshape(batch_size, num_curves * k, 2)

    return sampled_points


def load_init_circle_cubics(circle_svg_fp="./2719851_cubic_3_r0_diffvg_optm.svg", transform=None):

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        circle_svg_fp)

    for path in shapes:
        points = path.points
        num_control_points = path.num_control_points
        break

    # Transform points if applicable
    if transform:
        points = transform(points)

    # Truncate if sequence is too long
    len_points = points.shape[0]

    # Compute the cubics segments
    cubics = get_cubic_segments_from_points(
        points=points, num_control_points=num_control_points)

    desired_cubics_length = len_points // 3

    assert cubics.shape[0] == desired_cubics_length

    return cubics


# ----------------------------------------
def split_train_test(file_list, train_ratio=0.9):
    # split train and test
    random.shuffle(file_list)
    len_train = int(len(file_list)*train_ratio)
    file_list_train = file_list[:len_train]
    file_list_test = file_list[len_train:]

    return {
        "file_list_train": file_list_train,
        "file_list_test": file_list_test
    }


def save_filelist_csv(file_list, file_list_fp):
    file_list_df = pd.DataFrame(file_list, columns=["file_name"])
    file_list_df.to_csv(file_list_fp, index=False)


def split_trte_cmd_pts(svg_meta_fp, max_cmd_len_thresh=16, max_pts_len_thresh=53):
    # read meta file and calculate average length
    meta_data = pd.read_csv(svg_meta_fp)

    max_len_group_np = meta_data["max_len_group"]
    num_pts_np = meta_data["num_pts"]
    avg_len = max_len_group_np.mean()
    max_len = max_len_group_np.max()
    min_len = max_len_group_np.min()
    # Average length: 12.502676752255912
    print(f"Average length: {avg_len}")
    # Max length: 2681
    print(f"Max length: {max_len}")
    # Min length: 3
    print(f"Min length: {min_len}")

    cond_max_len = (max_len_group_np < max_cmd_len_thresh) & (
        num_pts_np < max_pts_len_thresh + 10)
    filtered_data = meta_data[cond_max_len]

    # If you want to extract the "id" column from the filtered data and add ".svg" extension:
    remove_long_file_list = (filtered_data["id"] + ".svg").tolist()
    # print("len_remove_long_file_list2: ", len(remove_long_file_list))
    # print("samples: ", remove_long_file_list[:10])

    # split train and test
    random.shuffle(remove_long_file_list)
    len_train = int(len(remove_long_file_list)*0.9)
    remove_long_file_list_train = remove_long_file_list[:len_train]
    remove_long_file_list_test = remove_long_file_list[len_train:]

    # save remove_long_file_list_train to csv
    remove_long_file_list_train_df = pd.DataFrame(
        remove_long_file_list_train, columns=["file_name"])
    remove_long_file_list_train_df.to_csv(
        "./dataset/file_list_train.csv", index=False)

    remove_long_file_list_test_df = pd.DataFrame(
        remove_long_file_list_test, columns=["file_name"])
    remove_long_file_list_test_df.to_csv(
        "./dataset/file_list_test.csv", index=False)


def split_trte_pts(svg_meta_fp, svg_data_dir, file_list_train_fp, file_list_test_fp, svg_data_img_dir="", max_pts_len_thresh=53, min_area_thresh=3000):
    # read meta file and calculate average length
    meta_data = pd.read_csv(svg_meta_fp)

    max_len_group_np = meta_data["max_len_group"]
    num_pts_np = meta_data["num_pts"]
    area_np = meta_data["area"]

    avg_len = num_pts_np.mean()
    max_len = num_pts_np.max()
    min_len = num_pts_np.min()
    # Average length: 12.502676752255912
    print(f"Average length: {avg_len}")
    # Max length: 2681
    print(f"Max length: {max_len}")
    # Min length: 3
    print(f"Min length: {min_len}")

    avg_area = area_np.mean()
    max_area = area_np.max()
    min_area = area_np.min()
    print(f"Average area: {avg_area}")
    print(f"Max area: {max_area}")
    print(f"Min area: {min_area}")

    # ----------------------------------------
    bins = np.arange(min_area, max_area, 500)
    # 面积直方图
    counts, edges = np.histogram(area_np, bins=bins)
    # for i in range(len(bins) - 1):
    #     bin_min = bins[i]
    #     bin_max = bins[i + 1] - 1
    #     print(f"{bin_min} and {bin_max}: {counts[i]}")
    # ----------------------------------------

    # -----------------------------------------------
    # 构造bins, 从min到max, 每隔10; max_len + 10 确保max_len包含在最后一个bin中
    # 10, 22
    bins = np.arange(min_len, min_len + 400, 15)
    # 计算直方图
    counts, edges = np.histogram(num_pts_np, bins=bins)
    sum_counts = np.sum(counts)

    small_svg_list_trte = []
    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1] - 1
        print(
            f"Processing Lengths between {bin_min} and {bin_max}: {counts[i]}")

        if counts[i] == 0:  # Skip empty bins
            continue

        # Create a new directory for the bin
        new_dir = svg_data_dir[:-1] + "_" + str(int(bin_max)) + "/"
        if (os.path.exists(new_dir)):
            shutil.rmtree(new_dir)
        os.makedirs(new_dir, exist_ok=True)

        new_img_dir = svg_data_dir[:-1] + "_" + str(int(bin_max)) + "_img/"
        if (os.path.exists(new_img_dir)):
            shutil.rmtree(new_img_dir)
        os.makedirs(new_img_dir, exist_ok=True)

        # delete new_dir
        shutil.rmtree(new_dir)
        shutil.rmtree(new_img_dir)
        # continue

        # Filter the data frame to contain only rows within the bin range
        filtered_data = meta_data[(num_pts_np >= bin_min) & (
            num_pts_np < bins[i + 1])]

        # Randomly sample up to 100 rows from the filtered data frame
        # sample_size = min(200, len(filtered_data))
        sample_size = len(filtered_data)
        st = 0 * sample_size
        ed = st + sample_size
        sampled_data = filtered_data[st:ed]
        # sampled_data = filtered_data.sample(n=sample_size, random_state=1)

        # Extract the "id" column from the sampled data and append ".svg" extension
        sampled_files = (sampled_data["id"] + ".svg").tolist()

        # Copy sampled files to the new directory
        for file in sampled_files:
            source_file = os.path.join(svg_data_dir, file)
            file_pre = file.split(".")[0]
            tmp_img_fp = os.path.join(svg_data_img_dir, file_pre + ".png")

            if os.path.exists(source_file) and os.path.exists(tmp_img_fp):
                destination_file = os.path.join(new_dir, file)
                # shutil.copy(source_file, destination_file)
                destination_file = os.path.join(new_img_dir, file_pre + ".png")
                # shutil.copy(tmp_img_fp, destination_file)
                small_svg_list_trte.append(file)

    # split train and test
    sep_res = split_train_test(small_svg_list_trte)
    file_list_train = sep_res["file_list_train"]
    file_list_test = sep_res["file_list_test"]

    # save remove_long_file_list_train to csv
    bin_ind = str(int(bins[1] - 1))
    file_list_train_img_fp = file_list_train_fp.replace(
        ".csv", "_" + bin_ind + ".csv")
    # save_filelist_csv(file_list_train, file_list_train_img_fp)
    file_list_test_img_fp = file_list_test_fp.replace(
        ".csv", "_" + bin_ind + ".csv")
    # save_filelist_csv(file_list_test, file_list_test_img_fp)

    # 计算直方图
    log_data = np.log1p(num_pts_np)  # 使用log1p来避免对数为0的情况
    counts, bins, patches = plt.hist(
        log_data, bins=30, edgecolor='k', alpha=0.65)

    # 标题和标签
    plt.title('Histogram of log1p(Lengths)')
    plt.xlabel('log1p(Length)')
    plt.ylabel('Frequency')

    plt.grid(axis='y', alpha=0.75)

    plt.savefig("./num_pts.png")

    # -----------------------------------------------

    cond_max_len = ((num_pts_np < max_pts_len_thresh)
                    & (area_np > min_area_thresh))
    filtered_data = meta_data[cond_max_len]

    # extract the "id" column from the filtered data and add ".svg" extension:
    remove_long_file_list = (filtered_data["id"] + ".svg").tolist()
    print("len_remove_long_file_list2: ", len(remove_long_file_list))
    # print("samples: ", remove_long_file_list[:10])

    # 只保留存在的文件
    existing_remove_long_file_list = []
    for file in remove_long_file_list:
        if os.path.exists(os.path.join(svg_data_dir, file)):
            append_flg = True
            if (len(svg_data_img_dir) > 0):
                file_pre = file.split(".")[0]
                if (not os.path.exists(os.path.join(svg_data_img_dir, file_pre + ".png"))):
                    append_flg = False

            if (append_flg):
                existing_remove_long_file_list.append(file)

    remove_long_file_list = existing_remove_long_file_list
    print("len_remove_long_file_list3: ", len(remove_long_file_list))

    # split train and test
    sep_res = split_train_test(remove_long_file_list)
    remove_long_file_list_train = sep_res["file_list_train"]
    remove_long_file_list_test = sep_res["file_list_test"]

    # save remove_long_file_list_train to csv
    save_filelist_csv(remove_long_file_list_train, file_list_train_fp)
    save_filelist_csv(remove_long_file_list_test, file_list_test_fp)
