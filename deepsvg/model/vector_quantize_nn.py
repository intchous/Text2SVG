import torch
from torch.autograd import Function
from torch import nn
# from .vector_quantize_pytorch import VectorQuantize as VectorQuantizePytorch

# from: https://github.com/pabloppp/pytorch-tools


class vector_quantize(Function):
    @staticmethod
    def forward(ctx, x, codebook):
        with torch.no_grad():
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            x_sqr = torch.sum(x ** 2, dim=1, keepdim=True)

            dist = torch.addmm(codebook_sqr + x_sqr, x,
                               codebook.t(), alpha=-2.0, beta=1.0)
            _, indices = dist.min(dim=1)

            ctx.save_for_backward(indices, codebook)
            ctx.mark_non_differentiable(indices)

            nn = torch.index_select(codebook, 0, indices)
            return nn, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors

            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output)

        return (grad_inputs, grad_codebook)


class binarize_fun(Function):
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        with torch.no_grad():
            binarized = (x > threshold).float()
            ctx.mark_non_differentiable(binarized)

            return binarized

    @staticmethod
    def backward(ctx, grad_output):
        grad_inputs = None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()

        return grad_inputs


class VectorQuantize(nn.Module):
    def __init__(self, embedding_size, k, ema_decay=0.99, ema_loss=False):
        """
        Takes an input of variable size (as long as the last dimension matches the embedding size).
        Returns one tensor containing the nearest neigbour embeddings to each of the inputs, 
        with the same size as the input, vq and commitment components for the loss as a touple 
        in the second output and the indices of the quantized vectors in the third: 
        quantized, (vq_loss, commit_loss), indices
        """
        super(VectorQuantize, self).__init__()

        self.codebook = nn.Embedding(k, embedding_size)
        self.codebook.weight.data.uniform_(-1./k, 1./k)
        self.vq = vector_quantize.apply

        self.ema_decay = ema_decay
        self.ema_loss = ema_loss
        if ema_loss:
            self.register_buffer('ema_element_count', torch.ones(k))
            self.register_buffer(
                'ema_weight_sum', torch.zeros_like(self.codebook.weight))

    def _laplace_smoothing(self, x, epsilon):
        n = torch.sum(x)
        return ((x + epsilon) / (n + x.size(0) * epsilon) * n)

    def _updateEMA(self, z_e_x, indices):
        mask = nn.functional.one_hot(
            indices, self.ema_element_count.size(0)).float()
        elem_count = mask.sum(dim=0)
        weight_sum = torch.mm(mask.t(), z_e_x)

        self.ema_element_count = (
            self.ema_decay * self.ema_element_count) + ((1-self.ema_decay) * elem_count)
        self.ema_element_count = self._laplace_smoothing(
            self.ema_element_count, 1e-5)
        self.ema_weight_sum = (
            self.ema_decay * self.ema_weight_sum) + ((1-self.ema_decay) * weight_sum)

        self.codebook.weight.data = self.ema_weight_sum / \
            self.ema_element_count.unsqueeze(-1)

    def idx2vq(self, idx, dim=-1):
        q_idx = self.codebook(idx)
        if dim != -1:
            q_idx = q_idx.movedim(-1, dim)
        return q_idx

    def forward(self, x, get_losses=True, dim=-1):
        if dim != -1:
            x = x.movedim(dim, -1)
        z_e_x = x.contiguous().view(-1, x.size(-1)) if len(x.shape) > 2 else x
        z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())
        vq_loss, commit_loss = None, None
        if self.ema_loss and self.training:
            self._updateEMA(z_e_x.detach(), indices.detach())
        # pick the graded embeddings after updating the codebook in order to have a more accurate commitment loss
        z_q_x_grd = torch.index_select(
            self.codebook.weight, dim=0, index=indices)
        if get_losses:
            vq_loss = (z_q_x_grd - z_e_x.detach()).pow(2).mean()
            commit_loss = (z_e_x - z_q_x_grd.detach()).pow(2).mean()

        z_q_x = z_q_x.view(x.shape)
        if dim != -1:
            z_q_x = z_q_x.movedim(-1, dim)
        return z_q_x, (vq_loss, commit_loss), indices.view(x.shape[:-1])


class Binarize(nn.Module):
    def __init__(self, threshold=0.5):
        """
        Takes an input of any size.
        Returns an output of the same size but with its values binarized (0 if input is below a threshold, 1 if its above)
        """
        super(Binarize, self).__init__()

        self.bin = binarize_fun.apply
        self.threshold = threshold

    def forward(self, x):
        return self.bin(x, self.threshold)


if __name__ == '__main__':
    # create a random tensor with 8 as its last dimension size
    e = torch.randn(1, 16, 16, 8)
    seq_len = 1
    z_dim = 256
    codebook_size = 1024
    e = torch.randn(140, seq_len, z_dim)
    print("e: ", e)

    # we create the module with embedding size of 8, a codebook of size 32 and make the codebook update using EMA
    # vquantizer = VectorQuantize(8, k=32, ema_loss=True)
    vquantizer = VectorQuantize(z_dim, k=codebook_size, ema_loss=True)
    # we quantize our tensor while also getting the loss components and the indices
    qe, (vq_loss, commit_loss), indices = vquantizer.forward(e)

    # NOTE While the model is in training mode, the codebook will always be updated when calling the forward method, in order to freeze the codebook for inference put it in evaluation mode with 'vquantizer.eval()'

    # NOTE 2 In order to update the module properly, add the loss components to the final model loss before calling backward(), if you set ema_loss to true you only need to add the commit_loss to the total loss, an it's usually multiplied by a value between 0.1 and 2, being 0.25 a good default value

    # loss = ... # whatever loss you have for your final output
    loss = 0.0
    loss += commit_loss * 0.25
    # loss += vq_loss # only if you didn't set the ema_loss to True
    print("commit_loss: ", commit_loss)
    # commit_loss:  tensor(0.9740)
    print("vq_loss: ", vq_loss)
    # vq_loss:  tensor(0.9740, grad_fn=<MeanBackward0>)
    # print("qe: ", qe)
    # qe的shape跟e的shape一样
    print("qe.shape: ", qe.shape)
    # qe.shape:  torch.Size([140, 1, 256])
    # print("indices: ", indices)
    print("indices.shape: ", indices.shape)
    # indices.shape:  torch.Size([140, 1])
    # indices.shape:  torch.Size([1, 16, 16])

    vquantizer_pytorch = VectorQuantizePytorch(
        dim=z_dim,
        codebook_size=codebook_size,
        # decay=0.8,
        # commitment_weight=0.,
        # use_cosine_sim=False,
    )

    quantized, indices, commit_loss = vquantizer_pytorch(e)
    print("commit_loss: ", commit_loss)
    # commit_loss:  tensor([0.9928], grad_fn=<AddBackward0>)

    # print("quantized: ", quantized)
    # print("quantized.shape: ", quantized.shape)
    # print("indices: ", indices)
    print("indices.shape: ", indices.shape)
    # indices.shape:  torch.Size([140, 1])
