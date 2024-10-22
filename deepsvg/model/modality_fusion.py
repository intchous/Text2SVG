import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


class SeqTxtFusion(nn.Module):
    def __init__(self, seq_latent_dim=512, txt_enc_dim=512, txt_latent_dim=512, mode='train'):
        super().__init__()
        self.mode = mode
        self.seq_latent_dim = seq_latent_dim
        self.txt_enc_dim = txt_enc_dim
        self.txt_latent_dim = txt_latent_dim

        # if (txt_latent_dim != 512):
        if (txt_latent_dim != txt_enc_dim):
            self.fc_text = nn.Linear(txt_enc_dim, txt_latent_dim)
        self.fc_text2seq = nn.Linear(txt_latent_dim, seq_latent_dim)
        # self.bottleneck = nn.Linear(seq_latent_dim + txt_latent_dim, seq_latent_dim)
        self.bottleneck = nn.Linear(seq_latent_dim * 2, seq_latent_dim)

    def forward(self, seq_feat, txt_emb):
        # 'p c b z' (S, G, N, z_dim)
        # seq_feat.shape:  torch.Size([1, 1, 512, 608])

        e_seq_emb = seq_feat

        if (self.txt_latent_dim != self.txt_enc_dim):
            l_txt_emb = self.fc_text(txt_emb)
        else:
            l_txt_emb = txt_emb
        # l_txt_emb.shape:  torch.Size([B, txt_latent_dim])

        l_txt_emb = self.fc_text2seq(l_txt_emb).unsqueeze(0).unsqueeze(0)
        # l_txt_emb.shape:  torch.Size([1, 1, 512, 608])

        feat_cat = torch.cat((e_seq_emb, l_txt_emb), -1)
        # feat_cat.shape:  torch.Size([1, 1, 512, 1216])

        fuse_z = self.bottleneck(feat_cat)
        # print("fuse_z.shape: ", fuse_z.shape)
        # fuse_z.shape:  torch.Size([1, 1, 512, 608])

        return fuse_z


class ModalityFusion(nn.Module):
    # img_size=64,  bottleneck_bits=512, seq_latent_dim=64, img_latent_dim=128, ngf=32, ref_nshot=4, mode='train'
    def __init__(self, bottleneck_bits=512, seq_latent_dim=64, img_enc_dim=1024, img_latent_dim=128,  mode='train'):
        super().__init__()
        self.mode = mode
        self.img_enc_fc = nn.Linear(img_enc_dim, img_latent_dim)
        self.bottleneck_bits = bottleneck_bits
        # self.ref_nshot = ref_nshot
        # self.fc_merge = nn.Linear(seq_latent_dim * opts.ref_nshot, 512)
        # n_downsampling = int(math.log(img_size, 2))
        # mult_max = 2 ** (n_downsampling)
        # the max multiplier for img feat channels is
        # self.fc_fusion = nn.Linear(ngf * mult_max + seq_latent_dim, opts.bottleneck_bits * 2, bias=True)
        self.fc_fusion = nn.Linear(
            seq_latent_dim + img_latent_dim, bottleneck_bits * 2, bias=True)

    def forward(self, seq_feat, img_feat):

        # seq_feat shape:  torch.Size([1, 50, 140, 256]) (S, G, N, z_dim)
        seq_feat_ = seq_feat.permute(
            2, 1, 0, *range(3, seq_feat.dim()))  # (N, G, S, z_dim)
        seq_feat_ = seq_feat_.contiguous().view(seq_feat_.size(0), seq_feat_.size(1),
                                                seq_feat_.size(2) * seq_feat_.size(3))
        # print("seq_feat_.shape1: ", seq_feat_.shape)
        # seq_feat_.shape1:  torch.Size([32, 52, 2048])
        # seq_feat_.shape1:  torch.Size([140, 1, 256])

        # seq_feat_ = self.fc_merge(seq_feat_)
        # print("seq_feat_.shape2: ", seq_feat_.shape)
        # seq_feat_.shape2:  torch.Size([32, 52, 512])

        seq_feat_cls = seq_feat_[:, 0]
        # print("seq_feat_cls.shape: ", seq_feat_cls.shape)
        # seq_feat_cls.shape:  torch.Size([32, 512])

        # img_feat.shape:  torch.Size([32, 1024])
        img_feat_enc = self.img_enc_fc(img_feat)
        # print("img_feat_enc.shape: ", img_feat_enc.shape)

        feat_cat = torch.cat((img_feat_enc, seq_feat_cls), -1)
        # print("feat_cat.shape: ", feat_cat.shape)
        # feat_cat.shape:  torch.Size([140, 1280])

        dist_param = self.fc_fusion(feat_cat)
        # print("dist_param.shape: ", dist_param.shape)
        # dist_param.shape:  torch.Size([140, 512])

        output = {}
        mu = dist_param[..., :self.bottleneck_bits]
        log_sigma = dist_param[..., self.bottleneck_bits:]

        # if self.mode == 'train':
        #     # calculate the kl loss and reparamerize latent code
        #     epsilon = torch.randn(*mu.size(), device=mu.device)
        #     z = mu + torch.exp(log_sigma / 2) * epsilon
        #     kl = 0.5 * torch.mean(torch.exp(log_sigma) +
        #                           torch.square(mu) - 1. - log_sigma)
        #     output['latent'] = z
        #     output['kl_loss'] = kl
        #     seq_feat_[:, 0] = z
        #     latent_feat_seq = seq_feat_

        # else:
        #     output['latent'] = mu
        #     output['kl_loss'] = 0.0
        #     seq_feat_[:, 0] = mu
        #     latent_feat_seq = seq_feat_

        epsilon = torch.randn(*mu.size(), device=mu.device)
        z = mu + torch.exp(log_sigma / 2) * epsilon
        # print("z.shape1: ", z.shape)
        # z.shape1:  torch.Size([140, 256])

        z = z.unsqueeze(1).unsqueeze(1)
        z = z.permute(2, 1, 0, *range(3, z.dim()))
        # print("z.shape2: ", z.shape)
        # z.shape2:  torch.Size([1, 1, 140, 256])
        mu = mu.unsqueeze(1).unsqueeze(1)
        mu = mu.permute(2, 1, 0, *range(3, mu.dim()))
        log_sigma = log_sigma.unsqueeze(1).unsqueeze(1)
        log_sigma = log_sigma.permute(2, 1, 0, *range(3, log_sigma.dim()))

        return z, mu, log_sigma
