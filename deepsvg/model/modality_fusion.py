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

        if (txt_latent_dim != txt_enc_dim):
            self.fc_text = nn.Linear(txt_enc_dim, txt_latent_dim)
        self.fc_text2seq = nn.Linear(txt_latent_dim, seq_latent_dim)
        self.bottleneck = nn.Linear(seq_latent_dim * 2, seq_latent_dim)

    def forward(self, seq_feat, txt_emb):
        # 'p c b z' (S, G, N, z_dim)
        e_seq_emb = seq_feat

        if (self.txt_latent_dim != self.txt_enc_dim):
            l_txt_emb = self.fc_text(txt_emb)
        else:
            l_txt_emb = txt_emb

        l_txt_emb = self.fc_text2seq(l_txt_emb).unsqueeze(0).unsqueeze(0)

        feat_cat = torch.cat((e_seq_emb, l_txt_emb), -1)

        fuse_z = self.bottleneck(feat_cat)

        return fuse_z


class ModalityFusion(nn.Module):
    def __init__(self, bottleneck_bits=512, seq_latent_dim=64, img_enc_dim=1024, img_latent_dim=128,  mode='train'):
        super().__init__()
        self.mode = mode
        self.img_enc_fc = nn.Linear(img_enc_dim, img_latent_dim)
        self.bottleneck_bits = bottleneck_bits
        self.fc_fusion = nn.Linear(
            seq_latent_dim + img_latent_dim, bottleneck_bits * 2, bias=True)

    def forward(self, seq_feat, img_feat):

        seq_feat_ = seq_feat.permute(
            2, 1, 0, *range(3, seq_feat.dim()))  # (N, G, S, z_dim)
        seq_feat_ = seq_feat_.contiguous().view(seq_feat_.size(0), seq_feat_.size(1),
                                                seq_feat_.size(2) * seq_feat_.size(3))

        seq_feat_cls = seq_feat_[:, 0]

        img_feat_enc = self.img_enc_fc(img_feat)

        feat_cat = torch.cat((img_feat_enc, seq_feat_cls), -1)

        dist_param = self.fc_fusion(feat_cat)

        output = {}
        mu = dist_param[..., :self.bottleneck_bits]
        log_sigma = dist_param[..., self.bottleneck_bits:]

        epsilon = torch.randn(*mu.size(), device=mu.device)
        z = mu + torch.exp(log_sigma / 2) * epsilon

        z = z.unsqueeze(1).unsqueeze(1)
        z = z.permute(2, 1, 0, *range(3, z.dim()))
        mu = mu.unsqueeze(1).unsqueeze(1)
        mu = mu.permute(2, 1, 0, *range(3, mu.dim()))
        log_sigma = log_sigma.unsqueeze(1).unsqueeze(1)
        log_sigma = log_sigma.permute(2, 1, 0, *range(3, log_sigma.dim()))

        return z, mu, log_sigma
