# Copyright (C) 2020 Denso IT Laboratory, Inc.
# All Rights Reserved

# Denso IT Laboratory, Inc. retains sole and exclusive ownership of all
# intellectual property rights including copyrights and patents related to this
# Software.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from skimage.segmentation._slic import _enforce_label_connectivity_cython


def conv_in_relu(in_c, out_c):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_c, out_c, 3, bias=False),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.ReLU()
    )


class CNNRIM(nn.Module):
    """
    code for
    T.Suzuki, ICASSP2020
    Superpixel Segmentation via Convolutional Neural Networks with Regularized Information Maximization
    https://arxiv.org/abs/2002.06765

    Args:
        in_c: int
            number of input channels. (5 indicates RGB+XY)
        n_spix: int
            number of superpixels
        n_filters: int
            number of filters in convolution filters.
            At i-th layer, output channels are n_filters * 2^{i+1}
        n_layers: int
            number of convolution layers
        use_recons: bool
            if True, use reconstruction loss for optimization
        use_last_inorm: bool
            if True, use instance normalization layer for output
    """
    def __init__(self, in_c=5, n_spix=100, n_filters=32, n_layers=5, use_recons=True, use_last_inorm=True):
        super().__init__()
        self.n_spix = n_spix
        self.use_last_inorm = use_last_inorm
        self.use_recons = use_recons
        out_c = n_spix
        if use_recons:
            out_c += 3

        layers = []
        for i in range(n_layers-1):
            layers.append(conv_in_relu(in_c, n_filters << i))
            in_c = n_filters << i
        layers.append(nn.Conv2d(in_c, out_c, 1))
        self.layers = nn.Sequential(*layers)
        if use_last_inorm:
            self.norm = nn.InstanceNorm2d(n_spix, affine=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):
        spix = self.layers(x)
        if self.use_recons:
            recons, spix = spix[:, :3], spix[:, 3:]
        else:
            recons = None
        if self.use_last_inorm:
            spix = self.norm(spix)
        return spix, recons


    def mutual_information(self, logits, coeff):
        """
        Mutual information defined in eq. (2)

        Args:
            logits: torch.Tensor
                A Tensor of shape (b, n, h, w)
            coeff: float
                corresponding to lambda in eq. (2)
        """
        prob = logits.softmax(1)
        pixel_wise_ent = - (prob * F.log_softmax(logits, 1)).sum(1).mean()
        marginal_prob = prob.mean((2, 3))
        marginal_ent = - (marginal_prob * torch.log(marginal_prob + 1e-16)).sum(1).mean()
        return pixel_wise_ent - coeff * marginal_ent
    

    def smoothness(self, logits, image):
        """
        Smoothness loss defined in eq. (3)

        Args:
            logits: torch.Tensor
                A Tensor of shape (b, n, h, w)
            image; torch.Tensor
                A Tensor of shape (b, c, h, w)
        """
        prob = logits.softmax(1)
        dp_dx = prob[..., :-1] - prob[..., 1:]
        dp_dy = prob[..., :-1, :] - prob[..., 1:, :]
        di_dx = image[..., :-1] - image[..., 1:]
        di_dy = image[..., :-1, :] - image[..., 1:, :]

        return (dp_dx.abs().sum(1) * (-di_dx.pow(2).sum(1)/8).exp()).mean() + \
            (dp_dy.abs().sum(1) * (-di_dy.pow(2).sum(1)/8).exp()).mean()


    def reconstruction(self, recons, image):
        """
        Reconstruction loss defined in eq. (4)

        Args:
            recons: torch.Tensor
                A Tensor of shape (b, c, h, w)
            image; torch.Tensor
                A Tensor of shape (b, c, h, w)
        """
        return F.mse_loss(recons, image)


    def __preprocess(self, image, device="cuda"):
        image = torch.from_numpy(image).permute(2, 0, 1).float()[None]
        h, w = image.shape[-2:]
        coord = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w))).float()[None]

        input = torch.cat([image, coord], 1).to(device)
        input = (input - input.mean((2, 3), keepdim=True)) / input.std((2, 3), keepdim=True)
        return input


    def optimize(self, image, n_iter=500, lr=1e-2, lam=2, alpha=2, beta=10, device="cuda"):
        """
        optimizer and generate superpixels

        Args:
            image: numpy.ndarray
                An array of shape (h, w, c)
            n_iter: int
                number of iterations for SGD
            lr: float
                learning rate
            lam: float
                used in eq. (2)
            alpha: float
                used in eq. (1)
            beta: float
                used in eq. (1)
            device: ["cpu", "cuda"]
        
        Return:
            spix: numpy.ndarray
                An array of shape (h, w)
        """
        input = self.__preprocess(image, device)
        optimizer = optim.Adam(self.parameters(), lr)

        for i in range(n_iter):
            spix, recons = self.forward(input)

            loss_mi = self.mutual_information(spix, lam)
            loss_smooth = self.smoothness(spix, input)
            loss = loss_mi + alpha * loss_smooth
            if recons is not None:
                loss = loss + beta * self.reconstruction(recons, input[:, :3])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[{i+1}/{n_iter}] loss {loss.item()}")

        return self.calc_spixel(image, device)


    def calc_spixel(self, image, device="cuda"):
        """
        generate superpixels

        Args:
            image: numpy.ndarray
                An array of shape (h, w, c)
            device: ["cpu", "cuda"]
        
        Return:
            spix: numpy.ndarray
                An array of shape (h, w)

        """
        input = self.__preprocess(image, device)
        spix, recons = self.forward(input)

        spix = spix.argmax(1).squeeze().to("cpu").detach().numpy()

        segment_size = spix.size / self.n_spix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        spix = _enforce_label_connectivity_cython(
            spix[None], min_size, max_size)[0]

        return spix


if __name__ == "__main__":
    import os
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt

    from skimage.segmentation import mark_boundaries


    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, type=str, help="/path/to/image")
    parser.add_argument("--n_spix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--n_filters", default=32, type=int, help="number of convolution filters")
    parser.add_argument("--n_layers", default=5, type=int, help="number of convolution layers")
    parser.add_argument("--lam", default=2, type=float, help="coefficient of marginal entropy")
    parser.add_argument("--alpha", default=2, type=float, help="coefficient of smoothness loss")
    parser.add_argument("--beta", default=2, type=float, help="coefficient of reconstruction loss")
    parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
    parser.add_argument("--n_iter", default=500, type=int, help="number of iterations")
    parser.add_argument("--out_dir", default="./", type=str, help="output directory")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNNRIM(5, args.n_spix, args.n_filters, args.n_layers).to(device)

    if args.image is None: # load sample image from scipy
        import scipy.misc
        img = scipy.misc.face()
    else:
        img = plt.imread(args.image)

    spix = model.optimize(img, args.n_iter, args.lr, args.lam, args.alpha, args.beta, device)

    plt.imsave(os.path.join(args.out_dir, "boundary.png"), mark_boundaries(img, spix))
    np.save("spixel", spix) # save generated superpixel as .npy file
