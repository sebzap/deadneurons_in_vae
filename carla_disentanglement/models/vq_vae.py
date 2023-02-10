import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.base import Flatten3D, View
from torch.utils.data import DataLoader
from typing import Callable
from collections.abc import Sequence
from datetime import datetime
from PIL.Image import Image as PILImage
from tqdm import tqdm
from carla_disentanglement.models.beta_vae import BetaVAE, VAEModule
from architectures.ResConv import Encoder, Decoder

# source https://nbviewer.jupyter.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA1D(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA1D, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), inputs)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings


class VQVAEModule(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0, num_channels=1):
        super(VQVAEModule, self).__init__()

        self._embedding_dim = embedding_dim

        self._encoder = Encoder(num_channels, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens,
                                num_channels)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return x_recon, loss, perplexity

    def encode(self, x: torch.Tensor):
        """Encodeds Input to latents

        Args:
            x (Tensor): input

        Returns:
            Tensor, Tensor: latent encoding, peplexity
        """
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        _, quantized, _, _ = self._vq_vae(z)
        return quantized

    def decode(self, z):
        """Decondes latent z into output

        Args:
            z (Tensor): latent

        Returns:
            Tensor: output
        """
        return self.decoder(z)

    def reconstruct(self, x):
        """Forward pass from input to reconstruction, without reparamterization trick.
        Only uses mu, ignores logvar
        Omits (sigmoid) activation for the use of BCEWithLogitsLoss

        Args:
            x (Tensor): input

        Returns:
            Tensor, Tensor: reconstruction, perplexity
        """
        x_recon, _, perplexity = self.forward(x)
        return x_recon, perplexity


class VQVAEModule1D(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(VQVAEModule1D, self).__init__()

        self._embedding_dim = embedding_dim

        self._encoder = Encoder(1, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq = nn.Sequential(
            Flatten3D(),
            nn.Linear(num_hiddens*16*16, 256),
            nn.ReLU(True),
            nn.Linear(256, embedding_dim)
        )
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA1D(num_embeddings, embedding_dim,
                                                commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, embedding_dim*16*16),                 # B, H
            nn.ReLU(True),
            View((-1, embedding_dim, 16, 16)),                 # B,  H,  16,  16
            Decoder(embedding_dim,
                    num_hiddens,
                    num_residual_layers,
                    num_residual_hiddens)
        )

    def forward(self, x):
        z = self._encoder(x)
        # print(z.shape)
        z = self._pre_vq(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return x_recon, loss, perplexity

    def encode(self, x: torch.Tensor):
        """Encodeds Input to latents

        Args:
            x (Tensor): input

        Returns:
            Tensor, Tensor: latent encoding, peplexity
        """
        z = self._encoder(x)
        z = self._pre_vq(z)
        _, quantized, _, _ = self._vq_vae(z)
        return quantized

    def decode(self, z):
        """Decondes latent z into output

        Args:
            z (Tensor): latent

        Returns:
            Tensor: output
        """
        return self._decoder(z)

    def reconstruct(self, x):
        """Forward pass from input to reconstruction, without reparamterization trick.
        Only uses mu, ignores logvar
        Omits (sigmoid) activation for the use of BCEWithLogitsLoss

        Args:
            x (Tensor): input

        Returns:
            Tensor, Tensor: reconstruction, perplexity
        """
        x_recon, _, perplexity = self.forward(x)
        return x_recon, perplexity


class VQVAE(BetaVAE):
    """
    https://arxiv.org/abs/1711.00937
    """

    def __init__(self, model: VQVAEModule, image_size=None):
        self.model = model
        super().__init__(model._encoder, model._decoder, reconstruction='mse')
        self.z_dim = self.model._embedding_dim
        self.image_size = image_size

    def initModel(self, encoder: nn.Module, decoder: nn.Module):
        pass

    def train(self, ds: torch.utils.data.Dataset,
              batch_size=256,
              epochs=100,
              loss_every=1,
              loss_callback: Callable[[Sequence], None] = None,
              eval_every=10,
              eval_callback: Callable[[Sequence, Sequence], None] = None,
              reconstruction_every=10,
              reconstruction_callback: Callable[[PILImage], None] = None,
              reconstruct_indices: Sequence = None,
              save_every=20,
              save_path="./checkpoints/",
              num_workers=8,
              data_variance=1.0):
        """Start training

        Args:
            ds (torch.utils.data.Dataset): Dataset
            batch_size (int, optional): batch size. Defaults to 256.
            epochs (int, optional): number of epoches. Defaults to 100.
            loss_every (int, optional): after how many epochs loss_callback will be called. Defaults to 1.
            loss_callback (Callable[[Sequence], None], optional): Callback function with [complete loss, reconstruction loss, kl loss]. Defaults to None.
            eval_every (int, optional): after how many epochs evaluation will be perforemd. Defaults to 10.
            eval_callback (Callable[[Sequence, Sequence], None], optional): Callback function with [complete loss, reconstruction loss, kl loss]. Defaults to None.
            reconstruction_every (int, optional): after how many epochs reconstruction_callback will be called. Defaults to 10.
            reconstruction_callback (Callable[[PILImage], None], optional): callback with reconstructed image. Defaults to None.
            save_every (int, optional): after how many epochs model will be saved. Defaults to 20.
            save_path (str, optional): path for model state dict. Defaults to "./checkpoints/".
        """

        train_size = int(0.75 * len(ds))
        test_size = len(ds) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size], generator=torch.Generator().manual_seed(0))

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        self.model.train()

        last_save = None

        with tqdm(range(epochs)) as bar:
            for epoch in bar:
                self.train_iter = epoch
                epoch_loss = torch.empty((len(data_loader), 3), dtype=torch.float32, device=self.device)
                for i, (x_true, _) in enumerate(data_loader):
                    x_true = x_true.to(self.device)

                    y, vq_loss, perplexity = self.model.forward(x_true)

                    recon_loss = self.reconstruction_loss(y, x_true).div(batch_size).div(data_variance)

                    loss = recon_loss + vq_loss

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=False)
                    self.optimizer.step()

                    epoch_loss[i, 0] = recon_loss.detach()
                    epoch_loss[i, 1] = vq_loss.detach()
                    epoch_loss[i, 2] = perplexity.detach()

                epoch_loss = epoch_loss.mean(axis=0).cpu().numpy()
                l1, l2, p = epoch_loss[0], epoch_loss[1], epoch_loss[2]

                self.losses.append([l1+l2, l1, l2, p])

                if epoch % save_every == 0:
                    last_save = "vae_" + str(epoch) + '_' + \
                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_"+str(l1+l2)+".save"
                    torch.save(self.model.state_dict(), save_path+last_save)

                bar.set_postfix(epoch_stat=epoch, reconstruction_loss=l1, vq_loss=l2, perplexity=p, last_save=last_save)

                if loss_callback:
                    if epoch % loss_every == 0:
                        loss_callback(self.losses)

                if eval_callback:
                    if epoch % eval_every == 0:
                        eval_loss, eval_scores = self.eval(test_dataset, batch_size=batch_size, data_variance=data_variance)
                        self.model.train()
                        self.eval_losses.append(eval_loss)
                        self.scores.append(eval_scores)
                        eval_callback(self.eval_losses, self.scores)

                if reconstruction_callback:
                    if epoch % reconstruction_every == 0:
                        reconstruction_callback(self.visualize_reconstruction(ds, indices=reconstruct_indices))

        if loss_callback:
            loss_callback(self.losses)

        if reconstruction_callback:
            reconstruction_callback(self.visualize_reconstruction(ds, indices=reconstruct_indices))

        last_save = "vae_" + \
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_"+str(l1+l2)+".save"
        torch.save(self.model.state_dict(), save_path+last_save)
        print("saved "+last_save)

    def eval(self, ds, batch_size=256, data_variance=1.0):
        self.model.eval()
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            epoch_loss = torch.empty((len(data_loader), 3), dtype=torch.float32, device=self.device)
            for i, (x_true, _) in enumerate(data_loader):
                x_true = x_true.to(self.device)

                y, vq_loss, perplexity = self.model.forward(x_true)
                recon_loss = self.reconstruction_loss(y, x_true).div(batch_size).div(data_variance)

                epoch_loss[i, 0] = recon_loss.detach()
                epoch_loss[i, 1] = vq_loss.detach()
                epoch_loss[i, 2] = perplexity.detach()

            epoch_loss = epoch_loss.mean(axis=0).cpu().numpy()
            l1, l2, p = epoch_loss[0], epoch_loss[1], epoch_loss[2]

            return [l1+l2, l1, l2, p], self.score(ds)

    def encode(self, x):
        return self.model.encode(x.to(self.device))
