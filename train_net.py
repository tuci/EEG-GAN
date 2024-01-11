import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from openpyxl import Workbook
from torch.utils.data import DataLoader
from Discriminator import *
from Generator import *
from Generator import Generator_WGAN_GP_Upsample_BigLinear_LayerNorm
from dataloader import data_generator, CustomRandomSamplerSlicedShuffled, CustomRandomBatchSamplerSlicedShuffled
from Train import *

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
torch.manual_seed(manualSeed)

def save_model(model,optimizer,epoch, path_to_model):
    state = {'epoch': epoch+1,
             'state_dict': model.state_dict(),
             'optim_dict' : optimizer.state_dict()
            }
    torch.save(state,path_to_model)


# define global parameters
#path_to_hdf5_file_train = "C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/hdf5_file_train_30files_chunking_shhs1.hdf5" # Root directory for dataset
#path_to_file_length_cumul="C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/trainFilesNum30secEpochsCumulative_30files_shhs1.pkl"
#path_to_file_length_train="C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/trainFilesNum30secEpochs_30files_shhs1.pkl"
# Root directory for dataset
# CHANGE DATA ROOT
path_to_hdf5_file_train = "hdf5_file_train_30files_chunking_shhs1.hdf5"
path_to_file_length_cumul="trainFilesNum30secEpochsCumulative_30files_shhs1.pkl"
path_to_file_length_train="trainFilesNum30secEpochs_30files_shhs1.pkl"

f_file_length_train = open(path_to_file_length_train, 'rb')
file_length_dic_train = pickle.load(f_file_length_train)
f_file_length_train.close()
f_file_length_cumul = open(path_to_file_length_cumul, 'rb')
f_file_length_dic_cumul = pickle.load(f_file_length_cumul)

workers = 0  # Number of workers for dataloader
batch_size = 128  # Batch size during training
# image_size = 64 # Spatial size of training images. All images will be resized to this size using a transformer.
nc = 1  # Number of channels in the training images. For color images this is 3
nz = 1  # Size of z latent vector (i.e. size of generator input)
ngf = 8  # Size of feature maps in generator
ndf = 8  # Size of feature maps in discriminator
num_epochs = 200 # Number of training epochs
lr = 1e-4  # DCGAN => 2e-4, WGAN_WC => 5e-5  WGAN_GP => 1e-4
beta1 = 0.5
beta2 = 0.999
ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
penalty_parameter = 10

# CHANGE SAVE PATHS
path_to_model_d = './modelD/WGAN_upsample_big_ln/'
path_to_model_g = './modelG/WGAN_upsample_big_ln/'
outfolder = './figure/WGAN_upsample_big_ln/'
path_to_results = './loss/WGAN_upsample_big_ln/loss_score_test.xlxs'
path_to_txt_D = './loss/WGAN_upsample_big_ln/loss_scoreD_test.txt'
path_to_txt_G = './loss/WGAN_upsample_big_ln/loss_scoreG_test.txt'


# Create the dataset
data_gen_train = data_generator(path_to_hdf5_file_train)
print("start dataloader train")
sampler = CustomRandomSamplerSlicedShuffled(path_to_hdf5_file_train, f_file_length_dic_cumul)
batch_sampler_random_shuffling = CustomRandomBatchSamplerSlicedShuffled(sampler, batch_size, file_length_dic_train)
data_iterator_Train = DataLoader(data_gen_train, batch_size=1, num_workers=workers,
                                 batch_sampler=batch_sampler_random_shuffling)


# define networks
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# discriminator
netD = Discriminator_DCGAN(ngpu).to(device) # CHANGE DISCRIMINATOR
if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))
# generator
netG = Generator_DCGAN().to(device) # CHANGE GENERATOR
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))

# define optimisers & loss function(if needed)
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
criterion = nn.BCELoss()

# train
# open save excel file
wb = Workbook()
sheet1 = wb.active
header1 = ['Epoch','LossD','ErrorFake','ErrorReal','GradientPenalty']
sheet1.append(header1)
sheet2 = wb.create_sheet('Generator Data')
header2 = ['Epoch', 'LossD', 'LossG']
sheet2.append(header2)
txt_D = open(path_to_txt_D, 'w')
txt_G = open(path_to_txt_D, 'w')

for epoch in range(num_epochs):
    print('Epoch [{}/{}]'.format(epoch, num_epochs))
    # CHANGE TRAIN FUNCTION
    scoresD, scoresG, real_data, fake_data = train_dcgan(data_iterator_Train, netD, netG, criterion,
                                                         optimizerD, optimizerG, device)
    # save models
    fileD = path_to_model_d + 'modelD_epoch{}.pt'.format(epoch)
    fileG = path_to_model_g + 'modelG_epoch{}.pt'.format(epoch)
    save_model(netD, optimizerD, epoch, fileD)
    save_model(netG, optimizerG, epoch, fileG)

    # plot signals
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(np.linspace(0, 29.992, 3750), real_data.cpu().detach().numpy().reshape(-1), 'r')
    ax2.plot(np.linspace(0, 29.992, 3750), fake_data.cpu().detach().numpy().reshape(-1), 'g')
    plt.savefig(outfolder + 'wgan_wc_test{}.png'.format(epoch))

    # store scores in an excel file
    # discriminator
    sheet1.append(scoresD)
    txt_D = open(path_to_txt_D, 'a')
    txt_D.write(str(scoresD) + '\n')
    txt_D.close()
    # generator
    sheet2.append(scoresG)
    txt_G = open(path_to_txt_G, 'a')
    txt_G.write(str(scoresG) + '\n')
    txt_G.close()
