import sys

# seeds 9863185, 1506004, 5863596, 9621873, 3972279, 7694365
# python project_train.py 5863596 False s5863596_ReLU
# python project_train.py 5863596 True s5863596_LeakyReLU

# python project_train.py 1506004 False s1506004_ReLU
# python project_train.py 1506004 True s1506004_LeakyReLU

# python project_train.py 9621873 False s9621873_ReLU
# python project_train.py 9621873 True s9621873_LeakyReLU

# python project_train.py 3972279 False s3972279_ReLU
# python project_train.py 3972279 True s3972279_LeakyReLU

# python project_train.py 7694365 False s7694365_ReLU
# python project_train.py 7694365 True s7694365_LeakyReLU


if len(sys.argv) < 4:
    print("need arguments")
    exit()

datatag=sys.argv[3]
seed = int(sys.argv[1])
leaky = sys.argv[2] == 'True' 

z_dim = 10
num_channels = 1
image_size = 64

epochs = 50
gamma=100.0

c_max = 50
c_epochs = 25

# %%
from carla_disentanglement.architectures.ProjectConv64 import GaussianConv64, Conv64Decoder
from carla_disentanglement.models.annealed_vae import AnnealedVAE
from carla_disentanglement.models.beta_vae import BetaVAE, VAEModule
import torch
import numpy as np
import random

# %%
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
from carla_disentanglement.datasets.dsprites import DSpritesDataset
ds = DSpritesDataset()

# %%
activation_layer = torch.nn.LeakyReLU if leaky else torch.nn.ReLU
tag_suffix = 'dsprites_deadNeuronTest_'+str(seed) 
if leaky:
    tag_suffix = tag_suffix + "_leaky"


module = VAEModule(GaussianConv64(z_dim, num_channels, image_size, activation_layer=activation_layer), Conv64Decoder(z_dim, num_channels, image_size, activation_layer=activation_layer))
vae = AnnealedVAE(module, gamma=gamma, max_c=c_max, iterations_c=len(ds)/256*c_epochs, reconstruction='bce', tag_suffix=tag_suffix)

# %%
#-0.0684 for seed 2
#-0.0855 for seed 1506004
for name, param in vae.model.named_parameters():
    if 'weight' in name:
        print("sample weight value", param[4,0,2,3])
        break

# %%
data = vae.train(ds, epochs=epochs, reconstruction_every=1, eval_every=1, score_every=10,
              save_every=10, batch_size=256, num_workers=0, reconstruct_indices=[0, 4300, 200000, -554, -20000, int(737280/2+500)], log_deadNeurons=True, log_stepdata=True)

# %%
import pickle

with open('./dndata/deadNeurondata.'+datatag+'.pkl', 'wb') as handle:
    pickle.dump(data['deadNeurondata'], handle, protocol=pickle.HIGHEST_PROTOCOL)

print("done")