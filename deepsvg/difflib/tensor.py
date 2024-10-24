from __future__ import annotations
import torch
import torch.utils.data
from typing import Union
from deepsvg.utils import int2bit
Num = Union[int, float]


class SVGTensor:
    COMMANDS_SIMPLIFIED = ["m", "l", "c", "EOS", "SOS"]
    CMD_ARGS_MASK = torch.tensor([[0, 0, 0, 0, 1, 1],   # m
                                  [0, 0, 0, 0, 1, 1],   # l
                                  [1, 1, 1, 1, 1, 1],   # c
                                  [0, 0, 0, 0, 0, 0],   # EOS
                                  [0, 0, 0, 0, 0, 0]])  # SOS

    class Index:
        COMMAND = 0
        START_POS = slice(1, 3)
        CONTROL1 = slice(3, 5)
        CONTROL2 = slice(5, 7)
        END_POS = slice(7, 9)

    class IndexArgs:
        CONTROL1 = slice(0, 2)
        CONTROL2 = slice(2, 4)
        END_POS = slice(4, 6)

    position_keys = ["control1", "control2", "end_pos"]
    all_position_keys = ["start_pos", *position_keys]
    arg_keys = position_keys
    all_arg_keys = ["start_pos", *arg_keys]
    cmd_arg_keys = ["commands", *arg_keys]
    all_keys = ["commands", *all_arg_keys]

    def __init__(self, commands, control1, control2, end_pos,
                 seq_len=None, label=None, PAD_VAL=0, ARGS_DIM=256, filling=0):

        self.commands = commands.reshape(-1, 1).float()

        self.control1 = control1.float()
        self.control2 = control2.float()
        self.end_pos = end_pos.float()

        self.seq_len = torch.tensor(
            len(commands)) if seq_len is None else seq_len
        self.label = label

        self.PAD_VAL = PAD_VAL
        self.ARGS_DIM = ARGS_DIM

        self.sos_token = torch.Tensor(
            [self.COMMANDS_SIMPLIFIED.index("SOS")]).unsqueeze(-1)
        self.eos_token = self.pad_token = torch.Tensor(
            [self.COMMANDS_SIMPLIFIED.index("EOS")]).unsqueeze(-1)

        self.filling = filling

    @property
    def start_pos(self):
        start_pos = self.end_pos[:-1]

        return torch.cat([
            start_pos.new_zeros(1, 2),
            start_pos
        ])

    @staticmethod
    def from_data(data, *args, **kwargs):
        return SVGTensor(data[:, SVGTensor.Index.COMMAND], data[:, SVGTensor.Index.CONTROL1],
                         data[:, SVGTensor.Index.CONTROL2], data[:, SVGTensor.Index.END_POS], *args, **kwargs)

    @staticmethod
    def from_cmd_args(commands, args, *nargs, **kwargs):
        return SVGTensor(commands, args[:, SVGTensor.IndexArgs.CONTROL1],
                         args[:, SVGTensor.IndexArgs.CONTROL2], args[:, SVGTensor.IndexArgs.END_POS], *nargs, **kwargs)

    def get_data(self, keys):
        return torch.cat([self.__getattribute__(key) for key in keys], dim=-1)

    @property
    def data(self):
        return self.get_data(self.all_keys)

    def copy(self):
        return SVGTensor(*[self.__getattribute__(key).clone() for key in self.cmd_arg_keys],
                         seq_len=self.seq_len.clone(), label=self.label, PAD_VAL=self.PAD_VAL, ARGS_DIM=self.ARGS_DIM,
                         filling=self.filling)

    def add_sos(self):
        self.commands = torch.cat([self.sos_token, self.commands])

        for key in self.arg_keys:
            v = self.__getattribute__(key)
            self.__setattr__(key, torch.cat(
                [v.new_full((1, v.size(-1)), self.PAD_VAL), v]))

        self.seq_len += 1
        return self

    def drop_sos(self):
        for key in self.cmd_arg_keys:
            self.__setattr__(key, self.__getattribute__(key)[1:])

        self.seq_len -= 1
        return self

    def add_eos(self):
        self.commands = torch.cat([self.commands, self.eos_token])

        for key in self.arg_keys:
            v = self.__getattribute__(key)
            self.__setattr__(key, torch.cat(
                [v, v.new_full((1, v.size(-1)), self.PAD_VAL)]))

        return self

    def pad(self, seq_len=51):
        pad_len = max(seq_len - len(self.commands), 0)

        self.commands = torch.cat(
            [self.commands, self.pad_token.repeat(pad_len, 1)])

        for key in self.arg_keys:
            v = self.__getattribute__(key)
            self.__setattr__(key, torch.cat(
                [v, v.new_full((pad_len, v.size(-1)), self.PAD_VAL)]))

        return self

    def unpad(self):
        # Remove EOS + padding
        for key in self.cmd_arg_keys:
            self.__setattr__(key, self.__getattribute__(key)[:self.seq_len])
        return self

    def draw(self, *args, **kwags):
        from deepsvg.svglib.svg import SVGPath
        return SVGPath.from_tensor(self.data).draw(*args, **kwags)

    def cmds(self):
        return self.commands.reshape(-1)

    def args(self, with_start_pos=False):
        if with_start_pos:
            return self.get_data(self.all_arg_keys)

        data = self.get_data(self.arg_keys)
        # data = data / 12 - 1  # data normalization, [0, 24] -> [-1, 1]
        return data

    def _get_real_commands_mask(self):
        mask = self.cmds() < self.COMMANDS_SIMPLIFIED.index("EOS")
        return mask

    def _get_args_mask(self):
        mask = SVGTensor.CMD_ARGS_MASK[self.cmds().long()].bool()
        return mask

    def get_relative_args(self):
        data = self.args().clone()

        real_commands = self._get_real_commands_mask()
        data_real_commands = data[real_commands]

        start_pos = data_real_commands[:-1,
                                       SVGTensor.IndexArgs.END_POS].clone()

        data_real_commands[1:, SVGTensor.IndexArgs.CONTROL1] -= start_pos
        data_real_commands[1:, SVGTensor.IndexArgs.CONTROL2] -= start_pos
        data_real_commands[1:, SVGTensor.IndexArgs.END_POS] -= start_pos
        data[real_commands] = data_real_commands

        mask = self._get_args_mask()
        data[mask] += self.ARGS_DIM - 1
        data[~mask] = self.PAD_VAL

        return data

    def get_binary_args(self):
        data = self.args().clone()
        data = int2bit(torch.clamp(data, 0, 255).int())  # {0, 1}
        data = data * 2 - 1  # {-1, 1}
        return data

    def sample_points(self, n=10):
        device = self.commands.device

        z = torch.linspace(0, 1, n, device=device)
        Z = torch.stack([torch.ones_like(z), z, z.pow(2), z.pow(3)], dim=1)

        Q = torch.tensor([
            [[0., 0., 0., 0.],  #  "m"
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],

            [[1., 0., 0., 0.],  # "l"
             [-1, 0., 0., 1.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],

            [[1., 0., 0., 0.],  #  "c"
             [-3, 3., 0., 0.],
             [3., -6, 3., 0.],
             [-1, 3., -3, 1.]],

            torch.zeros(4, 4),  # "a", no support yet

            torch.zeros(4, 4),  # "EOS"
            torch.zeros(4, 4),  # "SOS"
            torch.zeros(4, 4),  # "z"
        ], device=device)

        commands, pos = self.commands.reshape(-1).long(), self.get_data(
            self.all_position_keys).reshape(-1, 4, 2)
        inds = (commands == self.COMMANDS_SIMPLIFIED.index("l")) | (
            commands == self.COMMANDS_SIMPLIFIED.index("c"))
        commands, pos = commands[inds], pos[inds]

        Z_coeffs = torch.matmul(Q[commands], pos)

        # Last point being first point of next command, we drop last point except the one from the last command
        sample_points = torch.matmul(Z, Z_coeffs)
        sample_points = torch.cat(
            [sample_points[:, :-1].reshape(-1, 2), sample_points[-1, -1].unsqueeze(0)])

        return sample_points

    @staticmethod
    def get_length_distribution(p, normalize=True):
        start, end = p[:-1], p[1:]
        length_distr = torch.norm(end - start, dim=-1).cumsum(dim=0)
        length_distr = torch.cat([length_distr.new_zeros(1), length_distr])
        if normalize:
            length_distr = length_distr / length_distr[-1]
        return length_distr

    def sample_uniform_points(self, n=100):
        p = self.sample_points(n=n)

        distr_unif = torch.linspace(0., 1., n).to(p.device)
        distr = self.get_length_distribution(p, normalize=True)
        d = torch.cdist(distr_unif.unsqueeze(-1), distr.unsqueeze(-1))
        matching = d.argmin(dim=-1)

        return p[matching]
