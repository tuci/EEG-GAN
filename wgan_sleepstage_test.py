# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:12:40 2020

@author: PathakS
"""

import torch
from collections import OrderedDict
import sys, time
import h5py, pickle
import random, math
import torch
import os
from scipy import signal
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler
#from swd.swd import swd
from scipy.spatial.distance import euclidean
from openpyxl import Workbook
from random import shuffle
from Generator import *

torch.set_printoptions(threshold=5000)

class data_generator(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def __len__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances = f['data'].shape[0]
        f.close()
        print("Total Length instances in generator:", len_instances)
        return len_instances

    def __getitem__(self, idx):
        f = h5py.File(self.hdf5_file, 'r')
        x_30sec_epoch = f['data'][idx][:, 0, :].reshape(-1, 3750)
        y_30sec_epoch = f['label'][idx][0]
        f.close()
        return idx, x_30sec_epoch, y_30sec_epoch


class CustomRandomBatchSamplerSlicedShuffled(Sampler):
    def __init__(self, sampler, batch_size, file_length_dic):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        self.sampler = sampler
        self.batch_size = batch_size
        self.file_length_dic = file_length_dic

    def __iter__(self):
        batch = []
        len_tillCurrent = 0
        file_current = 0
        batch_no = 0
        for idx in self.sampler:
            batch.append(idx)
            len_tillCurrent = len_tillCurrent + 1
            if len(batch) == self.batch_size or len_tillCurrent == self.file_length_dic[str(file_current)]:
                batch_no = batch_no + 1
                #print("batch:", batch_no)
                # batch.sort()
                yield batch
                batch = []
            if len_tillCurrent == self.file_length_dic[str(file_current)]:
                len_tillCurrent = 0
                file_current = file_current + 1

    def __len__(self):
        length = np.sum(np.ceil(np.array(list(self.file_length_dic.values())) / self.batch_size), dtype='int32')
        print("length in batch sampler:", length)
        return length


class CustomRandomSamplerSlicedShuffled(Sampler):
    def __init__(self, hdf5_file, dic_length):
        self.hdf5_file = hdf5_file
        self.dic_length = dic_length

    def __iter__(self):
        i = 0
        end = 0
        lenRandomShuffle = 200
        f = h5py.File(self.hdf5_file, 'r')
        ModifiedIndices = np.where(f['label'][:].reshape(-1) == [2])[0]
        while end < len(ModifiedIndices):
            if i == 0:
                begin = 0
            else:
                begin = i * lenRandomShuffle
            i = i + 1
            if i * lenRandomShuffle > len(ModifiedIndices):
                end = len(ModifiedIndices)
            else:
                end = i * lenRandomShuffle
            slicedIndices = MutableSlice(ModifiedIndices, begin, end)
            # print(ModifiedIndices)
            random.shuffle(slicedIndices)
            # print(ModifiedIndices)
            # input("halt")

        iter_shuffledIndex = iter(ModifiedIndices)
        # input("halt")
        # iter_index=iter(torch.randperm(len_instances).tolist())
        f.close()
        return iter_shuffledIndex

    def __len__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances = np.where(f['label'][:].reshape(-1) == [2]).shape[0]
        f.close()
        return len_instances


class MutableSlice(object):
    def __init__(self, baselist, begin, end=None):
        self._base = baselist
        self._begin = begin
        self._end = len(baselist) if end is None else end

    def __len__(self):
        return self._end - self._begin

    def __getitem__(self, i):
        return self._base[self._begin + i]

    def __setitem__(self, i, val):
        self._base[i + self._begin] = val

def load_model(model, path):
    model_parameters = torch.load(path, map_location=device)
    model_weights = model_parameters['state_dict']
    state_dict_remove_module = OrderedDict()
    for k, v in model_weights.items():
        state_dict_remove_module[k] = v
    model.load_state_dict(state_dict_remove_module)
    return model.to(device)

def predict(model, path, data):
    model = load_model(model, path)
    model.eval()
    output = model(data)
    return output

def results_plot(readFile_df,readFile_df1,file_names):
    color=list(iter(plt.cm.tab20(np.linspace(0,1,16))))
    #color=plt.cm.orange(
    #print(color)
    #colormap = plt.cm.gist_ncar
    #plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
    f, axes=plt.subplots(2,5,sharex=True, figsize=(20,10))
    for i,files in enumerate(readFile_df):
        c1=color[i]
        c2=color[i+1]
        c3=color[i+2]
        #print(file_names[i])
        #print(files[0][1:])
        #print(files[1][1:])
        #input('halt')
        if file_names[i].strip('.xlsx')=='WGAN_WC':
            axes.flatten()[i].plot(files[0][1:],files[1][1:]/min(files[1][1:]),label='D', color=c1)
        else:
            axes.flatten()[i].plot(files[0][1:],files[1][1:]/max(files[1][1:]),label='D', color=c1)
        axes.flatten()[i].plot(files[0][1:],readFile_df1[i][1][1:]/max(readFile_df1[i][1][1:]),label='GP', color=c3)
        axes.flatten()[i].plot(files[0][1:],files[2][1:]/max(files[2][1:]),label='G', color=c2)
        axes.flatten()[i].set_title(file_names[i].strip('.xlsx'),fontsize=10)
        axes.flatten()[i].legend(loc='best')
    #plt.plot(df[0],df[2],'-r',label='Train Accuracy CNN')
    #plt.plot(df[0],df[4],'-b',label='Test Accuracy CNN')
    #plt.plot(df1[0],df1[2],'-g',label='Train Accuracy CNNLSTM')
    #plt.plot(df1[0],df1[4],'-y',label='Test Accuracy CNNLSTM')
    #plt.legend(loc='upper left')
    #plt.xticks(np.arange(1,200))
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    f.delaxes(axes.flatten()[9])
    f.text(0.52,0.04,'Epoch', ha='center',fontsize=13)
    f.text(0.09,0.5,'Loss', ha='center', rotation=90, fontsize=13)
    #plt.savefig('loss_scores_graph.pdf')  
    plt.show()


def frequency_signal(eeg,file_name):
    sfreq = 125
    f1, pxx1 = signal.periodogram(eeg, sfreq)
    plt.plot(f1, pxx1)
    print(file_name)
    plt.show()
    plt.savefig('./frequency_signal/'+file_name+'_freq.pdf')
    plt.close()

def euclidean_dist(real,fake):
    summ = 0
    #print(real.shape)
    #print(fake.shape)
    for i in range(len(real)):
        out = euclidean(real[i][0],fake[i][0])
        #print(out)
        summ = summ+out
    avg = summ/10
    return avg

def sw_dist(real, fake):
    summ = 0
    #print(real.shape)
    #print(fake.shape)
    for i in range(len(real)):
        out = wasserstein_distance(real[i][0],fake[i][0])
        summ = summ+out
    avg = summ/10
    return avg

if __name__ == '__main__': 
    
    #torch.set_printoptions(profile='full')
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #Initialization
    #path_to_trained_model = './modelG/upsample_ln_in_critic/modelG_epoch199.pt'
    path_to_hdf5_file_train = "C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/hdf5_file_train_30files_chunking_shhs1.hdf5" # Root directory for dataset
    path_to_file_length_cumul="C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/trainFilesNum30secEpochsCumulative_30files_shhs1.pkl"
    path_to_file_length_train="C:/Users/PathakS/OneDrive - Universiteit Twente/deepsleep/final_codes/SHHS/trainFilesNum30secEpochs_30files_shhs1.pkl"
    
    #path_to_hdf5_file_train = "hdf5_file_train_30files_chunking_shhs1.hdf5" # Root directory for dataset
    #path_to_file_length_cumul = "trainFilesNum30secEpochsCumulative_30files_shhs1.pkl"
    #path_to_file_length_train = "trainFilesNum30secEpochs_30files_shhs1.pkl"
    path_to_loss_scores='./ModelG/loss_scores/'
    path_to_results = 'distances.xlsx'
    f_file_length_train = open(path_to_file_length_train, 'rb')
    file_length_dic_train = pickle.load(f_file_length_train)
    f_file_length_train.close()
    f_file_length_cumul = open(path_to_file_length_cumul, 'rb')
    f_file_length_dic_cumul = pickle.load(f_file_length_cumul)
    b_size = 10
    ngpu = 1
    workers = 0
    num=0
    readFile_df=[]
    readFile_df1=[]

    # open excel for distance
    wb = Workbook()
    sheet1 = wb.active
    header1 = ['model_name','dist_real_fake', 'dist_real', 'dist_fake']
    sheet1.append(header1)
    sheet2 = wb.create_sheet('wasserstein distance')
    header2 = ['model_name','dist_real_fake', 'dist_real', 'dist_fake']
    sheet2.append(header2)

    # create networks
    networks = []
    netDCGAN = Generator_DCGAN(ngpu)
    netWGAN = Generator_WGAN_WC_L936(ngpu)
    netWGAN_DCNV_Norm = Generator_Deconv_Norm(ngpu)
    netWGAN_DCNV_Bi = Generator_Deconv_Bilinear(ngpu)
    netWGAN_DCNV_UP = Generator_Upsample_Bilinear(ngpu)
    netWGAN_UP_l4 = Generator_Upsample_bn_in_gen_l4(ngpu)
    net_WGAN_UP_l4_both = Generator_Upsample_bn_in_gen_l4(ngpu)
    netWGAN_UP_l5 = Generator_Upsample_bn_in_gen_l5(ngpu)
    netWGAN_UP_ln = Generator_Upsample_ln_in_critic_l4(ngpu)
    networks = np.array([netDCGAN, netWGAN, netWGAN_DCNV_Norm, netWGAN_DCNV_Bi,
                        netWGAN_DCNV_UP, net_WGAN_UP_l4_both, netWGAN_UP_l4,
                        netWGAN_UP_l5, netWGAN_UP_ln])

    # weights
    weight_DCGAN = './ModelG/weights/dcgan_normaldeconv_bn_gen.pt'
    weight_WGAN = './ModelG/weights/wgan_wc.pt'
    weight_WGAN_DCNV_Norm = './ModelG/weights/wgan_normaldeconv_bn_gen.pt'
    weight_WGAN_DCNV_Bi = './ModelG/weights/wgan_bilinear_bn_gen.pt'
    weight_WGAN_DCNV_UP = './ModelG/weights/wgan_upsample_bilinear_bn_gen.pt'
    weight_WGAN_UP_l4_both = './ModelG/weights/wgan_upsample_bn_in_both_epoch199.pt'
    weight_WGAN_UP_l4 = './ModelG/weights/wgan_upsample_l4.pt'
    weight_WGAN_UP_l5 = './ModelG/weights/wgan_upsample_l5.pt'
    weight_WGAN_UP_ln = './ModelG/weights/wgan_upsample_ln.pt'
    weigths = np.array([weight_DCGAN, weight_WGAN, weight_WGAN_DCNV_Norm, weight_WGAN_DCNV_Bi,
                       weight_WGAN_DCNV_UP, weight_WGAN_UP_l4_both, weight_WGAN_UP_l4,
                       weight_WGAN_UP_l5, weight_WGAN_UP_ln])

    # Create the dataset
    data_gen_train = data_generator(path_to_hdf5_file_train)
    print("start dataloader train")
    sampler = CustomRandomSamplerSlicedShuffled(path_to_hdf5_file_train, f_file_length_dic_cumul)
    batch_sampler_random_shuffling = CustomRandomBatchSamplerSlicedShuffled(sampler, b_size, file_length_dic_train)
    data_iterator_Train = DataLoader(data_gen_train, batch_size=1, num_workers=workers,
                                 batch_sampler=batch_sampler_random_shuffling)
    #data generation
    data_10 = next(iter(data_iterator_Train))
    #print(data_10[1])
    print(data_10[0])
    # shuffle real signals
    #idx=data_10[0]
    idx=torch.randperm(10)#,b_size,replace=False))
    #shuffle(idx)
    print(idx)
    shuffle_real = data_10[1][idx]#.view(data_10[1].size())
    #print(shuffle_real)
    #frequency_signal(shuffle_real[0].view(-1),'real_data1')
    #frequency_signal(shuffle_real[1].view(-1),'real_data2')
    #frequency_signal(data_10[0].view(-1))

    #noise creation
    noiseG = torch.empty((b_size, 1,160))
    noiseG = torch.nn.init.normal_(noiseG).to(device)
    
    #loss score excel files
    #files_loss_scores=os.listdir(path_to_loss_scores)
    #print(files_loss_scores)
    files_loss_scores=['WGAN_GP_Deconv_Bilinear.xlsx','WGAN_GP_Deconv_Normal.xlsx','WGAN_GP_Upsample_BigLinear_BN_Both.xlsx','WGAN_GP_Upsample_BigLinear_BN_Gen.xlsx','WGAN_GP_Upsample_BigLinear_LayerNorm.xlsx','WGAN_GP_Upsample_Deconv.xlsx','WGAN_GP_Upsample_SmallLinear_BN_Gen.xlsx']
    
    #prediction
    #for i in range(len(files_loss_scores)):
    for net, weight in zip(networks, weigths):
        print(weight)
        if weight.split('/')[-1]=='wgan_upsample_bn_in_both_epoch199.pt':
            print('Network: {}'.format(weight))
            if isinstance(net, Generator_DCGAN):
                noise = torch.empty((b_size, 1, 116))
                noise = torch.nn.init.normal_(noise).to(device)
            else:
                noise = noiseG
            output_signal = predict(net, weight, noise).cpu().detach()
    
            # shuffle generated signals
            #print(output_signal)
            shuffle_fake = output_signal[idx]
            #print(shuffle_fake)
    
            #plt.plot(np.linspace(0, 29.992, 3750), output_signal[0, :, :].cpu().detach().numpy().reshape(-1), 'r')
            #plt.show()
    
            # frequency signal
            frequency_signal(output_signal[0].view(-1),weight.split('/')[-1])
    
            # euclidean distance
            '''avg_distance_euc = euclidean_dist(data_10[1],output_signal)
            avg_distance_real_euc = euclidean_dist(data_10[1], shuffle_real)
            avg_distance_fake_euc = euclidean_dist(output_signal, shuffle_fake)
            
            # print distances
            print('--- Euclidean distance ---')
            print('Average distance between real and fake: {}\n'
              'Average distance between real signals: {}\n'
              'Average distance between generated signals: {}\n'.format(avg_distance_euc, avg_distance_real_euc,
                                                                        avg_distance_fake_euc))
            dist_euc = [weight.split('/')[-1], avg_distance_euc, avg_distance_real_euc, avg_distance_fake_euc]
            sheet1.append(dist_euc)
    
            # wasserstein distance
            avg_distance_was = sw_dist(data_10[1],output_signal)
            avg_distance_real_was = sw_dist(data_10[1], shuffle_real)
            avg_distance_fake_was = sw_dist(output_signal, shuffle_fake)
            # print distances
            print('--- Wasserstein distance ---')
            print('Average distance between real and fake: {}\n'
              'Average distance between real signals: {}\n'
              'Average distance between generated signals: {}\n'.format(avg_distance_was, avg_distance_real_was,
                                                                        avg_distance_fake_was))
            dist_was = [weight.split('/')[-1], avg_distance_was, avg_distance_real_was, avg_distance_fake_was]
            sheet2.append(dist_was)
            
            #plotting
            #if files_loss_scores[i].find('WGAN_GP'):
            readFile_df.append(pd.read_excel('./ModelG/loss_scores/'+files_loss_scores[num],sheet_name='Generator Data',index_col=None,header=None))
            readFile_df1.append(pd.read_excel('./ModelG/loss_scores/'+files_loss_scores[num],sheet_name='Sheet',index_col=None,header=None))
            #print(readFile_df)
            
            num+=1'''
    #input('halt')
    #wb.save(path_to_results)
    #results_plot(readFile_df,readFile_df1,files_loss_scores)