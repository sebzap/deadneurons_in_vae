"""
Implementaion for Beta VAE
"""
from distutils import log
from torch.utils import data
from carla_disentanglement.datasets.ground_truth_data import GroundTruthDataset
from carla_disentanglement.metrics.factor_vae_score import factor_vae_score
import os
import subprocess
from typing import Callable
from collections.abc import Sequence
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from PIL.Image import Image as PILImage
import numpy as np
from tqdm import tqdm
from carla_disentanglement.metrics.beta_vae_score import beta_vae_score
from torch.utils.tensorboard import SummaryWriter
from disentanglement_lib.evaluation.metrics import beta_vae
from disentanglement_lib.evaluation.metrics import factor_vae
from disentanglement_lib.evaluation.metrics import dci
from disentanglement_lib.evaluation.metrics import mig
from disentanglement_lib.evaluation.metrics import sap_score

import gin.tf
gin.parse_config_file("metrics.gin")


class VAEModule(nn.Module):
    """
    pytorch Module for Variational Auto Encoders
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, activation=nn.Sigmoid(), loss_activation: nn.Module = None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.activation = activation
        self.use_model_regulariser = False
        self.loss_activation = loss_activation if loss_activation is not None else self.activation

    def encode(self, x: torch.Tensor):
        """Encodeds Input to latents

        Args:
            x (Tensor): input

        Returns:
            Tensor: latent encoding
        """
        return self.encoder(x)

    def decode(self, z):
        """Decondes latent z into output (with (sigmoid) activation)

        Args:
            z (Tensor): latent

        Returns:
            Tensor: output
        """
        return self.activation(self.decoder(z))

    def reparametrize(self, mu, logvar):
        """Reparamterization trick for training

        Args:
            mu (Tensor): mean of latents
            logvar (Tensor): log variance of latents

        Returns:
            Tensor: latent sample
        """
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """Forward pass from input to reconstruction (including reparamterization trick).
        Omits (sigmoid) activation for the use of BCEWithLogitsLoss

        Args:
            x (Tensor): input

        Returns:
            Tensor, (Tensor, Tensor): reconstruction, (mu, log_var), z
        """
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        # NOTE: we use decodeR(!) here not decode (which includes sigmoid activation)
        return self.loss_activation(self.decoder(z)), (mu, logvar), z

    def reconstruct(self, x):
        """Forward pass from input to reconstruction, without reparamterization trick.
        Only uses mu, ignores logvar

        Args:
            x (Tensor): input

        Returns:
            Tensor, (Tensor, Tensor): reconstruction, (mu, log_var)
        """
        mu, logvar = self.encode(x)
        return self.activation(self.decoder(mu)), (mu, logvar)

    def eval_forward(self, x):
        """Forward pass from input to reconstruction, without reparamterization trick.
        Only uses mu, ignores logvar
        Omits (sigmoid) activation for the use of BCEWithLogitsLoss

        Args:
            x (Tensor): input

        Returns:
            Tensor, (Tensor, Tensor): reconstruction, (mu, log_var)
        """
        mu, logvar = self.encode(x)
        return self.loss_activation(self.decoder(mu)), (mu, logvar)

    def model_regulariser(self):
        return 0


class BetaVAE(object):
    """
    Beta VAE model
    https://openreview.net/forum?id=Sy2fzU9gl
    """

    def __init__(self, model: VAEModule, beta=1.0, reconstruction='bce', tag=None, model_regulariser_weight=0.01, tag_suffix: str = None):

        self.model = model
        if reconstruction == 'mse_no_activation':
            self.reconstruction_loss = nn.MSELoss(reduction='sum')
            self.model.loss_activation = nn.Identity()
        elif reconstruction == 'mse':
            self.reconstruction_loss = nn.MSELoss(reduction='sum')
        else:
            self.reconstruction_loss = nn.BCEWithLogitsLoss(reduction='sum')
            self.model.loss_activation = nn.Identity()

        self.num_channels = model.encoder._num_channels
        self.z_dim = model.encoder._latent_dim
        self.image_size = model.encoder._image_size

        self.beta = beta
        self.model_regulariser_weight = model_regulariser_weight

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.model.to(self.device)

        tag = str(tag)+'_' if tag else ''
        self.full_tag = tag+str(self.__class__.__name__)+self.getParamString()
        if tag_suffix is not None:
            self.full_tag += "_" + tag_suffix
        print(self.full_tag, tag_suffix)
        self.num_log_values = 4

        self.initOptimizer()

    def getParamString(self):
        return "_beta"+str(self.beta)

    def gauss_vae_regulariser(self, mu: torch.Tensor, log_var: torch.Tensor):
        """
        Compute the regularisation factor for training a VAE
        with a Gaussian latent space.

        Parameters
        ----------
        mu: Tensor
            Mean predictions from variational encoder.
        log_var: Tensor
            log_var predictions from variational encoder.

        Returns
        -------
        kl_div : Tensor
            The Kullback-Leibler divergence between a Gaussian distribution
            with the given parameters and the standard normal distribution.
        log info : None
        """
        return ((torch.exp(log_var) + mu*mu - log_var - 1)/2).sum(1).mean(0)

    def _visualize_reconstruction(self, ds: torch.utils.data.Dataset, count=10, indices=None):
        """Create visualatin of VAE reconstruction

        Args:
            ds (torch.utils.data.Dataset): dataset
            count (int, optional): Number of sampels to be visualized. Defaults to 10.

        Returns:
            torch.Tensor: image
        """

        if indices != None:
            ds = torch.utils.data.Subset(ds, indices)
            count = len(indices)

        data_loader = DataLoader(ds, batch_size=count)
        xs = next(iter(data_loader))[0]

        with torch.no_grad():
            # get reconstructions
            preds = self.model.reconstruct(xs.to(self.device))[0].cpu()

            # concatenate all data
            x_cat = torch.cat(
                [torch.cat([x, torch.zeros(self.num_channels, self.image_size, 1)], dim=-1) for x in xs], dim=-1)
            pred_cat = torch.cat(
                [torch.cat([x, torch.zeros(self.num_channels, self.image_size, 1)], dim=-1) for x in preds], dim=-1)

            return torch.cat([x_cat, pred_cat], dim=1)

    def visualize_reconstruction(self, ds: torch.utils.data.Dataset, count=10, indices=None):
        """Create visualatin of VAE reconstruction
        How to use: display(image, metadata={'width': '100%'})

        Args:
            ds (torch.utils.data.Dataset): dataset
            count (int, optional): Number of sampels to be visualized. Defaults to 10.

        Returns:
            PIL.Image: image
        """

        image = self._visualize_reconstruction(ds, count, indices)

        to_image = transforms.Compose([
            # inverts normalization of image # implement unnormalize when we use normalization
            # transforms.Normalize(-means / stds, 1. / stds),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            transforms.ToPILImage()
        ])

        return to_image(image)

    def visualize_latents_static(self, ds: torch.utils.data.Dataset, index=0, limit=3):
        image = self._visualize_latents_static(ds, index, limit=limit)

        to_image = transforms.Compose([
            # inverts normalization of image # implement unnormalize when we use normalization
            # transforms.Normalize(-means / stds, 1. / stds),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            transforms.ToPILImage()
        ])

        return to_image(image)

    def _visualize_latents_static(self, ds: torch.utils.data.Dataset, index=0, limit=3):
        inter = 2/3
        interpolation = torch.arange(-limit, limit+0.1, inter)

        img = ds.__getitem__(index)[0].cuda().unsqueeze(0)
        z_org = self.model.encode(img)[0]
        z_dim = z_org.shape[-1]
        if len(z_org.shape) < 2:
            z_org = z_org.unsqueeze(0)

        samples = []
        for z_index in range(z_dim):
            z = z_org.clone()
            inters = []
            for val in interpolation:
                z[:, z_index] = val
                sample = self.model.decode(z).data
                inters.append(sample)
                inters.append(torch.full((1, sample.shape[1], sample.shape[2], 1), .5).cuda())

            samples.append(torch.cat(inters, -1))
            samples.append(torch.full((1, sample.shape[1], 1, samples[-1].shape[-1]), .5).cuda())

        samples = torch.cat(samples, -2).cpu()
        return samples.squeeze()

    def visualize_latents(self, ds: torch.utils.data.Dataset, index=0, output_dir='./viz/', delay=100):
        limit = 3
        inter = 2/3
        interpolation = torch.arange(-limit, limit+0.1, inter)

        img = ds.__getitem__(index)[0].cuda().unsqueeze(0)
        z_org = self.model.encode(img)[0]
        z_dim = z_org.shape[-1]
        if len(z_org.shape) < 2:
            z_org = z_org.unsqueeze(0)

        samples = []
        for z_index in range(z_dim):
            z = z_org.clone()
            for val in interpolation:
                z[:, z_index] = val
                sample = self.model.decode(z).data
                samples.append(sample)

        samples = torch.cat(samples).cpu()
        gif = samples.view(z_dim, len(interpolation), self.num_channels, self.image_size, self.image_size).transpose(0, 1)

        for j in range(gif.shape[0]):
            save_image(tensor=gif[j].cpu(),
                       fp=os.path.join(output_dir, '{}.png'.format(j)),
                       nrow=z_dim, pad_value=1)

        image_str = os.path.join(output_dir, '*.png')
        output_gif = os.path.join(output_dir, self.full_tag+'_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.gif')
        str1 = 'magick convert -delay '+str(delay)+' -loop 0 ' + image_str + ' ' + output_gif
        subprocess.call(str1, shell=True)

        return output_gif

    def initOptimizer(self, lr=1e-4, betas=None):
        """Initialize adam optimizer

        Args:
            lr (float, optional): Learning rate. Defaults to 1e-4.
            betas ((float,float), optional): betas param for Adam.
        """
        if betas is not None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def beforeTrain(self):
        # keep track of training
        self.writer = SummaryWriter(
            "runs/"+self.full_tag+"_"+datetime.now().strftime('%b%d_%H-%M-%S'))

    def afterTrainStep(self, epoch_data, epoch_step):
        pass

    def train(self, ds: torch.utils.data.Dataset,
              batch_size=256,
              epochs=100,
              eval_every=10,
              score_every=10,
              reconstruction_every=10,
              reconstruct_indices: Sequence = None,
              save_every=20,
              save_path="./checkpoints/",
              num_workers=8,
              score_ds: GroundTruthDataset = None,
              max_iter=None,
              log_deadNeurons=False,
              log_stepdata=False):
        """Start training

        Args:
            ds (torch.utils.data.Dataset): Dataset
            batch_size (int, optional): batch size. Defaults to 256.
            epochs (int, optional): number of epoches. Defaults to 100.
            score (int, optional): after how many epochs score metrics will be calculated. Defaults to 10.
            eval_every (int, optional): after how many epochs evaluation will be perforemd. Defaults to 10.
            reconstruction_every (int, optional): after how many epochs reconstruction_callback will be called. Defaults to 10.
            save_every (int, optional): after how many epochs model will be saved. Defaults to 20.
            save_path (str, optional): path for model state dict. Defaults to "./checkpoints/".
        """

        self.beforeTrain()

        train_size = int(0.75 * len(ds))
        test_size = len(ds) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size], generator=torch.Generator().manual_seed(0))

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        self.global_iter = 0
        self.model.train()

        last_save = None

        if log_deadNeurons:
            dn_weights = []
            dn_update_matix = []
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    dn_weights.append(name+" "+str(param.shape)[11:-1])
                    # two update maps: num updates, last update

                    dn_update_matix.append([torch.zeros(param.shape), torch.zeros(param.shape), torch.zeros([epochs] + list(param.shape),dtype=torch.bool)])

        for epoch in range(epochs):
            self.epoch_start = self.global_iter
            self.train_iter = epoch
            epoch_data = torch.empty((len(data_loader), self.num_log_values), dtype=torch.float32, device=self.device)

            if log_deadNeurons:
                dn_epoch_data = torch.empty((len(data_loader), len(dn_weights)), dtype=torch.float32, device=self.device)

            with tqdm(data_loader) as bar:
                for i, (x_true, _) in enumerate(bar):
                    self.global_iter += 1

                    x_true = x_true.to(self.device)

                    loss = self.calc_loss(x_true, batch_size, epoch_data, i)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=False)
                    if log_deadNeurons:
                        dn_i=0
                        for name, param in self.model.named_parameters():
                            if 'weight' in name:
                                dn_filter = (param.grad == 0)
                                dn_epoch_data[i, dn_i] = dn_filter.sum().div(param.grad.numel())
                                if log_stepdata:
                                    self.writer.add_scalar('Dead Neurons_/weight.'+str(dn_i)+'.'+name+" "+str(param.shape)[11:-1], dn_epoch_data[i, dn_i].cpu().numpy(), self.global_iter)
                                
                                dn_filter_inv = ~dn_filter
                                update_map, iteration_map, deadneuron_map_epochs = dn_update_matix[dn_i]
                                update_map[dn_filter_inv] +=1 #update_map
                                iteration_map[dn_filter_inv] = self.global_iter #iteration_map
                                deadneuron_map_epochs[epoch, dn_filter] = True

                                dn_i += 1

                    self.optimizer.step()

                    self.afterTrainStep(epoch_data, i)

                avg_loss = self.log_data(epoch, epoch_data, 'train')
                if log_deadNeurons:
                    dn_epoch_data = dn_epoch_data.detach().mean(axis=0).cpu().numpy()
                    for dn_i, name in enumerate(dn_weights):
                        self.writer.add_scalar('Dead Neurons/weight.'+str(dn_i)+'.'+name, dn_epoch_data[dn_i], epoch)
                        temp = ((dn_update_matix[dn_i][1] < self.epoch_start)*1.).mean().detach().cpu().numpy()
                        self.writer.add_scalar('Dead Neurons/weight.'+str(dn_i)+'.'+name+"_", temp, epoch)

                        self.writer.add_scalar('Dead_Neurons/weight.'+str(dn_i)+'.'+name, (dn_update_matix[dn_i][2][epoch]*1.0).mean(), epoch)


                if epoch % save_every == 0 and epoch > 0:
                    last_save = self.full_tag + "_" + str(epoch) + '_' + \
                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_"+str(avg_loss)+".save"
                    torch.save(self.model.state_dict(), save_path+last_save)

                print({"epoch": epoch, "step": self.global_iter, "loss": avg_loss, "last_save": last_save})
                # bar.set_postfix(epoch_stat=epoch, global_iter=self.global_iter, loss=avg_loss, last_save=last_save)

            do_eval = epoch % eval_every == 0
            do_recon = epoch % reconstruction_every == 0
            do_score = epoch % score_every == 0
            if do_eval or do_recon or do_score:
                self.model.eval()

            if do_eval:
                eval_loss = self.eval(test_dataset, batch_size=batch_size)
                self.log_data(epoch, eval_loss, 'eval')

            if do_recon:
                img = self._visualize_reconstruction(ds, indices=reconstruct_indices)
                self.writer.add_image('Reconstruction', torch.clamp(img, 0, 1), epoch)

            if do_score:
                scores = self.score(score_ds if score_ds is not None else ds)
                print(scores)
                for score_name, score_values in scores.items():
                    # self.writer.add_scalars(score_name, score_values, global_step=epoch)
                    for score_value_name, value in score_values.items():
                        self.writer.add_scalar(score_name+"/"+score_value_name, value, global_step=epoch)

            if do_eval or do_recon or do_score:
                self.model.train()

            if max_iter and self.global_iter >= max_iter:
                break

        # if we didn't calc scores in last iteration do it now
        if not do_score:
            scores = self.score(score_ds if score_ds is not None else ds)
            print(scores)
            for score_name, score_values in scores.items():
                # self.writer.add_scalars(score_name, score_values, global_step=epoch)
                for score_value_name, value in score_values.items():
                    self.writer.add_scalar(score_name+"/"+score_value_name, value, global_step=epoch)

        last_save = self.full_tag + "_" + \
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_"+str(avg_loss)+".save"
        torch.save(self.model.state_dict(), save_path+last_save)
        print("saved "+last_save)

        return_dict = {}
        if log_deadNeurons:
            return_dict['deadNeurondata'] = dn_update_matix
            return_dict['weights'] = dn_weights

        return return_dict

    def score(self, ds):
        random_state = np.random.RandomState(0)
        beta_vae_score = beta_vae.compute_beta_vae_sklearn(
            ds, make_representor(self),
            random_state, batch_size=64, num_train=10000, num_eval=5000)

        random_state = np.random.RandomState(0)
        factor_vae_score = factor_vae.compute_factor_vae(
            ds, make_representor(self),
            random_state, num_variance_estimate=10000)

        random_state = np.random.RandomState(0)
        mig_score = mig.compute_mig(ds, make_representor(self), random_state)

        # random_state = np.random.RandomState(0)
        # dci_score = dci.compute_dci(ds, make_representor(self), random_state)

        random_state = np.random.RandomState(0)
        sap = sap_score.compute_sap(ds, make_representor(self), random_state)

        return {
            "beta_vae_score": beta_vae_score,
            "factor_vae_score": factor_vae_score,
            "mig_score": mig_score,
            # "dci_score": dci_score,
            "sap_score": sap,
        }

    def calc_loss(self, x_true, batch_size, epoch_data, epoch_step):
        y, (mu, log_var), _ = self.model.forward(x_true)
        kld = self.gauss_vae_regulariser(mu, log_var)
        reg_loss = self.beta * kld
        recon_loss = self.reconstruction_loss(y, x_true).div(batch_size)

        loss = recon_loss + reg_loss

        if self.model.use_model_regulariser:
            loss += self.model_regulariser_weight * self.model.model_regulariser()

        epoch_data[epoch_step, 0] = recon_loss.detach()
        epoch_data[epoch_step, 1] = reg_loss.detach()
        epoch_data[epoch_step, 2] = kld.detach()

        with torch.no_grad():
            mse = nn.functional.mse_loss(
                self.model.activation(y) if isinstance(self.model.loss_activation, nn.Identity) else y,
                x_true,
                reduction='sum').div(batch_size)
            epoch_data[epoch_step, 3] = mse.detach()

        return loss

    def log_data(self, epoch, epoch_data, type='train'):
        epoch_data = epoch_data.detach().mean(axis=0).cpu().numpy()
        l1, l2 = epoch_data[0], epoch_data[1]

        self.writer.add_scalar('Loss reconstruction/'+type, l1, epoch)
        self.writer.add_scalar('Loss regularisation/'+type, l2, epoch)
        self.writer.add_scalar('KL divergence/'+type, epoch_data[2], epoch)
        self.writer.add_scalar('MSE reconstruction/'+type, epoch_data[3], epoch)

        return l1 + l2

    def eval(self, ds, batch_size=256):
        self.model.eval()
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            epoch_loss = torch.empty((len(data_loader), self.num_log_values), dtype=torch.float32, device=self.device)
            for i, (x_true, _) in enumerate(tqdm(data_loader)):
                x_true = x_true.to(self.device)

                self.calc_loss(x_true, batch_size, epoch_loss, i)

            return epoch_loss

    def loadModel(self, checkpoint, folder="./checkpoints/"):
        """Loads model checkpint

        Args:
            checkpoint (str): Filename of checkpoint
            folder (str, optional): dir where checkpoint is located. Defaults to "./checkpoints/".
        """
        self.model.load_state_dict(torch.load(folder + checkpoint))
        self.model.eval()

    def encode(self, x):
        return self.model.encode(x.to(self.device))[0]


def make_representor(model: BetaVAE):
    """
    Encloses the pytorch ScriptModule in a callable that can be used by `disentanglement_lib`.
    Parameters
    ----------
    model : torch.nn.Module or torch.jit.ScriptModule
        The Pytorch model.
    cuda : bool
        Whether to use CUDA for inference. Defaults to the return value of the `use_cuda`
        function defined above.
    Returns
    -------
    callable
        A callable function (`representation_function` in dlib code)
    """
    # Deepcopy doesn't work on ScriptModule objects yet:
    # https://github.com/pytorch/pytorch/issues/18106
    # model = deepcopy(model)
    # cuda = use_cuda() if cuda is None else cuda
    # model = model.cuda() if cuda else model.cpu()

    # Define the representation function
    def _represent(x):
        # assert isinstance(x, np.ndarray), \
        #     "Input to the representation function must be a ndarray."
        # assert x.ndim == 4, \
        #     "Input to the representation function must be a four dimensional NHWC tensor."
        # # Convert from NHWC to NCHW
        # x = np.moveaxis(x, 3, 1)
        # # Convert to torch tensor and evaluate
        # x = torch.from_numpy(x).float().to('cuda' if cuda else 'cpu')
        with torch.no_grad():
            y = model.encode(x)
        y = y.cpu().numpy()
        assert y.ndim == 2, \
            "The returned output from the representor must be two dimensional (NC)."
        return y

    return _represent
