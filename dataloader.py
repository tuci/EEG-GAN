import numpy as np
import math, h5py, random
from torch.utils.data import Dataset, Sampler


class data_generator(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def __len__(self):
        f = h5py.File(self.hdf5_file, 'r')
        len_instances = f['data'].shape[0]
        f.close()
        #print("Total Length instances in generator:", len_instances)
        return len_instances

    def __getitem__(self, idx):
        f = h5py.File(self.hdf5_file, 'r')
        x_30sec_epoch = data_normalizer(f['data'][idx][:, 0, :].reshape(-1, 3750))
        y_30sec_epoch = f['label'][idx][0]
        if y_30sec_epoch == 1:
            y_30sec_epoch = 0
        elif y_30sec_epoch == 3:
            y_30sec_epoch = 1
        elif y_30sec_epoch == 4:
            y_30sec_epoch = 2
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
        ModifiedIndices = np.where(f['label'][:].reshape(-1) == [4])[0]
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

def data_normalizer(data_30secs):
    data_ch_mean=(np.mean(data_30secs,axis=1))
    data_ch_std=(np.std(data_30secs,axis=1))
    if math.isnan(data_ch_std) or not data_ch_std:
        data_ch_norm=(data_30secs-data_ch_mean)
    else:
        data_ch_norm=(data_30secs-data_ch_mean)/data_ch_std

    #x=np.array([data_ch_norm_list])
    return data_ch_norm