from einops import rearrange
import torch

from deepsvg.utils.utils import _unpack_group_batch, _make_seq_first, _make_batch_first

from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .basic_blocks import FCN_args, ResNet

from .vector_quantize_nn import VectorQuantize

from .image_encoder import ImageEncoder
from .image_decoder import ImageDecoder
from .modality_fusion import ModalityFusion

from .config import _DefaultConfig
from .utils import (_get_args_padding_mask,
                    _get_args_key_padding_mask, _generate_square_subsequent_mask)


class SVGEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len):
        super().__init__()

        self.cfg = cfg

        self.embed_fcn = nn.Linear(cfg.n_args, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(
            cfg.d_model, max_len=seq_len, dropout=cfg.dropout)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.embed_fcn.weight, mode="fan_in")

    def forward(self, args):
        S, GN, p_n = args.shape
        src_arg_embed = self.embed_fcn(args)
        src = self.pos_encoding(src_arg_embed)
        return src


class ConstEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len):
        super().__init__()

        self.cfg = cfg

        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(
            cfg.d_model, max_len=seq_len, dropout=cfg.dropout)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.cfg.d_model))
        return src


class ModifiedConstEmbedding(nn.Module):
    def __init__(self, cfg: _DefaultConfig, seq_len):
        super().__init__()

        self.cfg = cfg
        self.seq_len = seq_len
        self.PE = PositionalEncodingLUT(
            cfg.d_model, max_len=seq_len, dropout=cfg.dropout)

        self.z_to_dmodel = nn.Linear(cfg.dim_z, cfg.d_model)

    def forward(self, z):
        N = z.size(1)

        z_transformed = self.z_to_dmodel(z).repeat(
            self.seq_len, 1, 1)

        # 位置编码
        position_encoded = self.PE(z.new_zeros(
            self.seq_len, N, self.cfg.d_model))

        # 结合z_transformed和位置编码
        src = z_transformed + position_encoded

        return src


class Encoder(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.cfg = cfg

        seq_len = cfg.max_enc_len if hasattr(
            cfg, 'max_enc_len') else cfg.max_total_len

        self.embedding = SVGEmbedding(cfg, seq_len)
        dim_label = None

        encoder_layer = TransformerEncoderLayerImproved(
            cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, cfg.n_layers, encoder_norm)

        self.enc_fc = nn.Linear(cfg.d_model*seq_len, cfg.d_model)

    def forward(self, args, label=None):
        S, G, N, p_n = args.shape
        l = None

        args = args.reshape(args.size(0), args.size(1) *
                            args.size(2), *args.shape[3:])

        padding_mask = _get_args_padding_mask(args, seq_dim=0)
        key_padding_mask = _get_args_key_padding_mask(args, seq_dim=0)

        src = self.embedding(args)

        memory = self.encoder(
            src, mask=None, src_key_padding_mask=key_padding_mask, memory2=l)

        # ------------------------------------------------------
        if (self.cfg.avg_path_zdim):

            # 1. 对 memory 和 padding_mask 进行元素乘法
            masked_memory = memory * padding_mask

            # 2. 转换维度
            transposed_memory = masked_memory.permute(
                1, 0, 2)  # 形状: [batch_size, seq_len, d_model]

            # 3. 调整形状 [batch_size, d_model * seq_len]
            flattened_memory = transposed_memory.reshape(
                transposed_memory.size(0), -1)

            z = self.enc_fc(flattened_memory).unsqueeze(0)

        else:
            z = memory * padding_mask
        # ------------------------------------------------------
        z = _unpack_group_batch(N, z)
        z_trans = z.transpose(0, 1)
        return z_trans


class VAE(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(VAE, self).__init__()

        self.enc_mu_fcn = nn.Linear(cfg.d_model, cfg.dim_z)
        self.enc_sigma_fcn = nn.Linear(cfg.d_model, cfg.dim_z)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.enc_mu_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_mu_fcn.bias, 0)
        nn.init.normal_(self.enc_sigma_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_sigma_fcn.bias, 0)

    def forward(self, z):
        mu, logsigma = self.enc_mu_fcn(z), self.enc_sigma_fcn(z)
        sigma = torch.exp(logsigma / 2.)
        z = mu + sigma * torch.randn_like(sigma)

        return z, mu, logsigma


class Bottleneck(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Linear(cfg.d_model, cfg.dim_z)

    def forward(self, z):
        return self.bottleneck(z)


class Decoder(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(Decoder, self).__init__()

        self.cfg = cfg
        dim_label = None

        if cfg.pred_mode == "autoregressive":
            self.embedding = SVGEmbedding(cfg, cfg.max_total_len)

            square_subsequent_mask = _generate_square_subsequent_mask(
                self.cfg.max_total_len+1)
            self.register_buffer("square_subsequent_mask",
                                 square_subsequent_mask)
        else:  # "one_shot"

            seq_len = cfg.max_dec_len if hasattr(
                cfg, 'max_dec_len') else cfg.max_total_len

            if hasattr(cfg, 'ModifiedConstEmbedding') and cfg.ModifiedConstEmbedding:
                self.embedding = ModifiedConstEmbedding(cfg, seq_len)
            else:
                self.embedding = ConstEmbedding(cfg, seq_len)

            if cfg.args_decoder:
                self.argument_embedding = ConstEmbedding(cfg, seq_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(
            cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout, d_global2=dim_label)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, cfg.n_layers_decode, decoder_norm)

        if cfg.rel_targets:
            args_dim = 2 * cfg.args_dim
        if cfg.bin_targets:
            args_dim = 8
        else:
            args_dim = cfg.args_dim + 1

        self.fcn = FCN_args(cfg.d_model, cfg.n_args, args_dim, cfg.abs_targets)

        self.use_sigmoid = cfg.use_sigmoid

    def _get_initial_state(self, z):
        hidden, cell = torch.split(torch.tanh(
            self.fc_hc(z)), self.cfg.d_model, dim=2)
        hidden_cell = hidden.contiguous(), cell.contiguous()
        return hidden_cell

    def forward(self, z,  args, label=None):

        N = z.size(2)
        if args is not None:
            S = args.size(0)
            G = args.size(1)

        l = self.label_embedding(label).unsqueeze(
            0) if self.cfg.label_condition else None

        if self.cfg.connect_through:
            z = rearrange(z, 'p c b d -> c (p b) d')

        if self.cfg.pred_mode == "autoregressive":

            args = args.reshape(args.size(0), args.size(
                1) * args.size(2), *args.shape[3:])
            src = self.embedding(args)

            key_padding_mask = _get_args_key_padding_mask(args, seq_dim=0)

            out = self.decoder(
                src, z, tgt_mask=self.square_subsequent_mask[:S, :S], tgt_key_padding_mask=key_padding_mask, memory2=l)

        else:  # "one_shot"

            # TODO: ConstEmbedding, src 是与z无关的纯位置编码? 是否改成ModifiedConstEmbedding
            src = self.embedding(z)

            out = self.decoder(src, z, tgt_mask=None,
                               tgt_key_padding_mask=None, memory2=l)

        args_logits = self.fcn(out)

        # TODO: 是否需要加sigmoid?
        if self.use_sigmoid:
            args_logits = torch.sigmoid(args_logits)

        reshape_out_logits = args_logits.reshape(
            args_logits.size(0), -1, N, *args_logits.shape[2:])

        return reshape_out_logits


class SVGTransformer(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super(SVGTransformer, self).__init__()

        self.cfg = cfg
        self.encoder = Encoder(cfg)

        if cfg.use_resnet:
            self.resnet = ResNet(cfg.d_model)

        if cfg.use_vae:
            self.vae = VAE(cfg)
        else:
            self.bottleneck = Bottleneck(cfg)
            self.encoder_norm = LayerNorm(
                cfg.dim_z, elementwise_affine=False)

        if cfg.use_vqvae:
            self.vqvae = VectorQuantize(
                embedding_size=cfg.vq_edim,
                k=cfg.codebook_size,
                ema_loss=True
            )

        self.decoder = Decoder(cfg)

        if cfg.use_model_fusion:
            self.img_encoder = ImageEncoder(
                img_size=cfg.img_size, input_nc=1, ngf=16, norm_layer=nn.LayerNorm)
            self.img_decoder = ImageDecoder(
                img_size=cfg.img_size, input_nc=cfg.dim_z, output_nc=1, ngf=16, norm_layer=nn.LayerNorm)

            self.modality_fusion = ModalityFusion(
                bottleneck_bits=cfg.dim_z, seq_latent_dim=cfg.d_model, img_enc_dim=cfg.d_img_model, img_latent_dim=cfg.img_latent_dim)

    def forward(self, args_enc, args_dec, ref_img=None, label=None, z=None, return_tgt=True, encode_mode=False,  return_indices=False):

        args_enc = _make_seq_first(args_enc)  # Possibly None, None
        args_dec_ = _make_seq_first(args_dec)

        flg_enc = False
        if z is None:
            flg_enc = True
            z = self.encoder(args_enc, label)

            if self.cfg.use_resnet:
                z = self.resnet(z)

            if self.cfg.use_vae:
                z, mu, logsigma = self.vae(z)

            elif self.cfg.use_model_fusion:
                # image encoding
                img_encoder_out = self.img_encoder(ref_img)
                img_feat = img_encoder_out['img_feat']  # bs, ngf * (2 ** 6)

                # modality funsion
                z, mu, logsigma = self.modality_fusion(z, img_feat)

            else:
                z = self.encoder_norm(self.bottleneck(z))

            if self.cfg.use_vqvae:
                batch_size, max_num_groups = z.shape[2], z.shape[0]

                z = rearrange(z, 'p c b z -> b (p c) z')

                # 将 z reshape
                z_reshaped = z.view(
                    z.shape[0], z.shape[1], -1, self.cfg.vq_edim)
                # tokenization
                # quantized, indices, commit_loss = self.vqvae.forward(z)
                quantized, (vq_loss, commit_loss), indices = self.vqvae.forward(
                    z_reshaped)
                quantized = quantized.view(z.shape[0], z.shape[1], -1)

                if return_indices:
                    return indices

                z = rearrange(quantized, 'b (p c) z -> p c b z',
                              p=max_num_groups)

        else:
            z = _make_seq_first(z)

        if encode_mode:
            return z

        # batch first
        self.latent_z = z.permute(2, 1, 0, *range(3, z.dim()))

        out_logits = self.decoder(z, args_dec_, label)

        out_logits_batch_first = out_logits.permute(
            2, 1, 0, *range(3, out_logits.dim()))

        res = {
            "args_logits": out_logits_batch_first
        }

        if self.cfg.use_model_fusion:
            # image decoding
            img_feat_ = z.permute(2, 1, 0, *range(3, z.dim()))

            img_feat_ = img_feat_.contiguous().view(img_feat_.size(0), img_feat_.size(1),
                                                    img_feat_.size(2) * img_feat_.size(3))

            img_feat_z = img_feat_[:, 0]

            img_decoder_out = self.img_decoder(
                img_feat_z, trg_img=ref_img)

            rec_img = img_decoder_out['gen_imgs']

            res["rec_img"] = rec_img

        if return_tgt:
            res["tgt_args"] = args_dec

            if (flg_enc):
                if self.cfg.use_vae:
                    res["mu"] = _make_batch_first(mu)
                    res["logsigma"] = _make_batch_first(logsigma)

                elif self.cfg.use_model_fusion:
                    rec_img_l1loss = img_decoder_out['img_l1loss']
                    res["rec_img_l1loss"] = rec_img_l1loss

                    res["mu"] = _make_batch_first(mu)
                    res["logsigma"] = _make_batch_first(logsigma)

                if self.cfg.use_vqvae:
                    res["vqvae_loss"] = commit_loss

        return res
