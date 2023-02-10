from carla_disentanglement.datasets.dsprites import DSpritesDataset, DSpritesDatasetMini
import numpy as np
from models.annealed_vae import AnnealedVAE
from models.beta_tcvae import BetaTCVAE
from models.beta_vae import BetaVAE
from models.info_vae import InfoVAE
from architectures.SimpleConv64 import *
from architectures.Conv64 import *
from architectures.ResConv import *
from architectures.ResConv64 import *
from architectures.LeakyConv64 import *
from architectures.LConv64 import *
from architectures.LSConv64 import *
from architectures.LSConv64Stride1 import *
import torch
import random
ds = DSpritesDataset()

seed = 2
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

z_dim = 10
num_channels = 1
image_size = 64

vae = BetaVAE(GaussianLConv64(z_dim, num_channels, image_size), LConv64Decoder(z_dim, num_channels, image_size))

vae.initOptimizer(lr=5e-4)  # , betas=(0.9,0.999)

vae.train(ds, epochs=1, max_iter=1.5e6, reconstruction_every=1, eval_every=1, batch_size=64,
          num_workers=0, reconstruct_indices=[0, 4300, 200000, -554, -20000, 737280/2+500])
