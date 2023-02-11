import sys
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import argparse
from glob import glob
import os
from tqdm import tqdm
import matlab


class FileCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = {}
        self.access_queue = []

    def _evict_oldest(self):
        oldest_file = self.access_queue.pop(0)
        self.cache.pop(oldest_file)

    def get(self, file_path):
        if file_path in self.cache:
            self.access_queue.remove(file_path)
            self.access_queue.append(file_path)
            return self.cache[file_path]
        else:
            return None

    def put(self, file_path, content):
        if len(self.cache) == self.max_size:
            self._evict_oldest()
        if file_path in self.cache:
            self.access_queue.remove(file_path)
        self.cache[file_path] = content
        self.access_queue.append(file_path)


class DSTLDataset(Dataset):
    def __init__(self,
                 protocols: list,  # this list will contain the folder/protocols names to include in the dataset
                 slice_len: int,    # this will define the slice size of each signal that will be input to the model
                 ds_type='train',  # either 'train' or 'test'
                 slice_overlap_ratio=0.5,   # this is the overlap ratio for each slice generated from a signal
                                            # this value will affect the number of slices that is possible to create from each signal
                 ds_path='/home/mauro/Research/DSTL/DSTL_DATASET_1_0',
                 file_postfix='',
                 noise_model='AWGN', snr_dbs=[30], seed=None,
                 apply_noise=True, apply_wchannel=None,
                 override_gen_map=False,
                 normalize=False,
                 transform=None, target_transform=None):

        self.protocols = protocols
        self.slice_len = slice_len
        self.slice_overlap_ratio = slice_overlap_ratio
        self.overlap = int(self.slice_len * self.slice_overlap_ratio)
        self.noise_model = noise_model
        self.snr_dbs = snr_dbs
        self.seed = seed
        if not (self.seed is None):
            np.random.seed(self.seed)


        assert(ds_type in ['train', 'test'])
        self.ds_type = ds_type
        if file_postfix != '' and file_postfix[-1] != '_':
            file_postfix += '_'
        info_filename = 'ds_info__'+file_postfix+str(len(protocols))+'class.pkl'
        ds_info_path = os.path.join(ds_path, info_filename)
        do_gen_info = True
        if os.path.exists(ds_info_path) and (not override_gen_map):
            ans = input('File '+info_filename+' already exists. Do you wanna create a new one? [y/n]')
            if ans.lower() in ['n', 'no']:
                do_gen_info = False

        if do_gen_info and (not override_gen_map):
            self.ds_info = self.generate_ds_map(ds_path, info_filename)
        else:
            self.ds_info = pickle.load(open(ds_info_path, 'rb'))

        if normalize:
            #Todo
            pass

        # let's assign inputs and labels
        self.transform = transform
        self.target_transform = target_transform
        self.apply_wchannel = apply_wchannel
        self.apply_noise = apply_noise

        self.signal_cache = FileCache(max_size=20e3)

        # let's initialize Matlab engine
        if not (self.apply_wchannel is None):
            self.possible_channels = ['TGn', 'TGax', 'Rayleigh', 'relative']
            assert(self.apply_wchannel in self.possible_channels)
            self.channel_map = {}
            from dstl.preprocessing.matutils import matutils
            self.mateng = matutils.MatlabEngine()  # todo check if we need any custom paths
            # initialize each channel object for each protocol used
            self.chan_models = {}
            for ix, p in enumerate(protocols):
                if p == '802_11n':
                    tgn = self.mateng.eng.wlanTGnChannel('SampleRate', float(20e6), 'DelayProfile', 'Model-B', 'LargeScaleFadingEffect', 'Pathloss', 'PathGainsOutputPort', True)
                    self.chan_models[ix] = tgn
                    self.channel_map['TGn'] = ix
                elif p == '802_11ax':
                    tgax = self.mateng.eng.wlanTGaxChannel('SampleRate', float(20e6), 'ChannelBandwidth', 'CBW20', 'DelayProfile', 'Model-B', 'LargeScaleFadingEffect', 'Pathloss', 'PathGainsOutputPort', True)
                    self.chan_models[ix] = tgax
                    self.channel_map['TGax'] = ix
                elif p == '802_11b':
                    rayleighB = self.mateng.eng.comm.RayleighChannel('SampleRate', float(11e6), 'PathDelays', float(1.5e-9), 'AveragePathGains', float(-3), 'PathGainsOutputPort', True)
                    self.chan_models[ix] = rayleighB
                    self.channel_map['RayleighB'] = ix
                elif (p == '802_11b_upsampled') or (p == '802_11g'):
                    rayleigh = self.mateng.eng.comm.RayleighChannel('SampleRate', float(20e6), 'PathDelays', float(1.5e-9), 'AveragePathGains', float(-3), 'PathGainsOutputPort', True)
                    self.chan_models[ix] = rayleigh
                    self.channel_map['Rayleigh'] = ix

    def generate_ds_map(self, ds_path, filename, test_ratio=0.2):
        examples_map = {}
        class_map = dict(zip(self.protocols, range(len(self.protocols))))
        # retrieve the list of signals (.mat) from every folder/protocol specified
        for i, p in enumerate(self.protocols):
            path = os.path.join(ds_path, p)
            if os.path.isdir(path):
                mat_list = sorted(glob(os.path.join(path, '*.mat')))
                examples_map[p] = dict(
                    zip(
                        list(range(len(mat_list))),
                        mat_list
                    )
                )
            else:
                sys.exit('[DSTLDataset] folder ' + path + ' not found. Aborting...')

        # now let's go through each class examples and assign a global sample index
        # based on the slice len and overlap configuration

        # also, we need to assign a unique slice index to each possible slice to be used with __getitem__ function
        # and assign the corresponding label
        data_ixs = {}
        labels_ixs = {}
        ixs_count = 0
        for c in tqdm(examples_map.keys(), desc='Analyzing signal dataset...'):
            for ix, path in examples_map[c].items():
                sig = sio.loadmat(path)
                len_sig = sig['waveform'].shape[0]
                window_ixs = list(range(0, len_sig-self.slice_len, self.overlap))
                n_windows = len(window_ixs)
                examples_map[c][ix] = {'slices': window_ixs,
                                       'path': examples_map[c][ix]}  # here we modify the original content
                for w in window_ixs:
                    data_ixs[ixs_count] = {'path': path, 'slice_ix': w}
                    labels_ixs[ixs_count] = class_map[c]
                    ixs_count += 1


        # lastly, we separate the training and testing dataset by randomly sampling a certain % of data samples indexes
        test_data_ixs = {}
        test_labels_ixs = {}
        test_ixs = np.random.choice(ixs_count, size=int(ixs_count*test_ratio), replace=False)
        for i in test_ixs.tolist():
            i_data = data_ixs.pop(i)
            test_data_ixs[i] = i_data
            i_label = labels_ixs.pop(i)
            test_labels_ixs[i] = i_label
        # we also need to store a map from linear indexes (used by ray when retrieving data from dataset)
        train_ixs = dict(zip(range(len(data_ixs.keys())), list(data_ixs.keys())))
        test_ixs = dict(zip(range(len(test_data_ixs.keys())), list(test_data_ixs.keys())))


        # after creating class map and examples map, let's store this information for future use
        ds_info_path = os.path.join(ds_path, filename)
        ds_info = {
            'class_map': class_map, 'examples_map': examples_map,
            'ds_indexes':
                {
                    'train': {'data': data_ixs, 'labels': labels_ixs},
                    'test': {'data': test_data_ixs, 'labels': test_labels_ixs}
                },
            'ixs_maps': {'train': train_ixs, 'test': test_ixs}  # linear indexes maps (used by ray/torch)
            }
        pickle.dump(ds_info, open(ds_info_path, 'wb'))

        return ds_info

    def info(self):
        ds_info = {
            'slice_len': self.slice_len,
            'numsamps': {'train' : len(self.ds_info['ds_indexes']['train']['data'].keys()),
                         'test': len(self.ds_info['ds_indexes']['test']['data'].keys())},
            'nclasses': len(self.protocols),
            'nchans': 2,    # real and imag components are separated on different channels
        }
        return ds_info

    def __len__(self):
        return len(self.ds_info['ds_indexes'][self.ds_type]['data'].keys())

    def __getitem__(self, idx):
        dataset = self.ds_info['ds_indexes'][self.ds_type]
        # first retrieve the internal sample index relative to the linear index idx
        s_idx = self.ds_info['ixs_maps'][self.ds_type][idx]
        # retrieve the info relative to this input sample
        obs_info = dataset['data'][s_idx]
        # let's first retrieve the signal from the cache (or load it in if not present)
        sig_dict = self.signal_cache.get(obs_info['path'])
        self.last_file_loaded = obs_info['path']
        if sig_dict is None:
            mat_dict = sio.loadmat(obs_info['path'])
            self.signal_cache.put(obs_info['path'], {'np': mat_dict['waveform'], 'mat': self.mateng.py2mat_array(mat_dict['waveform'])})
            sig_dict = self.signal_cache.get(obs_info['path'])

        label = dataset['labels'][s_idx]
        # apply wireless channel and noise if required
        chan_sig = self.apply_wchan(sig_dict['mat'], label) if not (self.apply_wchannel is None) else sig_dict['np']
        noisy_sig = self.apply_AWGN(chan_sig) if self.apply_noise else chan_sig

        # then, retireve the relative slice of the requested dataset sample
        obs = noisy_sig[obs_info['slice_ix']:obs_info['slice_ix']+self.slice_len, 0]
        obs = np.stack((obs.real, obs.imag))

        if self.transform:
            obs = self.transform(obs)
        if self.target_transform:
            label = self.target_transform(label)
        return obs, label

    # Function to change the shape of obs
    # the input is obs with shape (channel, slice)
    def chan2sequence(self, chan):
        seq = np.empty((chan.size))
        seq[0::2] = chan[0]
        seq[1::2] = chan[1]
        return seq

    def apply_AWGN(self, sig):
        # apply AWGN noise based on the levels specified when instantiating the dataset object
        # if more than one is specified, a random level is picked among the one specified
        # first, let's compute signal power (in Watts) as rms(sig)**2
        rms = np.sqrt(np.mean(np.abs(sig) ** 2))
        sig_W = rms ** 2  # power of signal in Watts
        # convert signal power in dBW
        SNR = np.random.choice(self.snr_dbs)
        sig_dbW = 10 * np.log10(sig_W / 1)
        # now compute the relative noise power based on the specified SNR level
        noise_dbW = sig_dbW - float(SNR)
        # noise variance = noise power
        noise_var = 10 ** (noise_dbW / 10)
        # obtain noise standard deviation
        noise_std = np.sqrt(noise_var)
        # now generate the noise samples.
        # NOTE: since we are generating ** complex ** noise samples, the nominal variance for a
        # normal complex distribution is equal to 1/2 instead of 1.
        # https://en.wikipedia.org/wiki/Complex_normal_distribution
        # var = std**2 = 1/2 ---> std = sqrt(1/2) = 1/sqrt(2)
        complex_std = noise_std * 1 / np.sqrt(2)
        noise_samples = np.random.normal(0, complex_std, size=sig.shape) + 1j * np.random.normal(0, complex_std,
                                                                                                 size=sig.shape)
        # now apply noise to the signal
        noisy_sig = sig + noise_samples
        return noisy_sig

    def apply_wchan(self, mat_sig, label):
        if self.apply_wchannel == 'relative':
            # in this case we apply the channel relative to the protocol used
            channel = self.chan_models[label]
        else:
            # in other cases, we have a fixed channel we want to apply to all the signals
            # in this case, 802.11b is not supported in its native sampling rate (11 MHz) due to consistency with
            # sampling rate of other standards (20 MHz)
            assert(not ('802_11b' in self.protocols))
            chan_ix = self.channel_map[self.apply_wchannel]
            channel = self.chan_models[chan_ix]

        proc_sig = self.mateng.eng.step(channel, mat_sig, nargout=1)
        return np.array(proc_sig)



if __name__ == "__main__":
    # myds = DSTLDataset(['802_11ax', '802_11b', '802_11n', '802_11g'], slice_len=128, slice_overlap_ratio=0.5) # case with mixed sampling rates and only AWGN
    myds = DSTLDataset(['802_11ax', '802_11b_upsampled', '802_11n', '802_11g'], slice_len=128, slice_overlap_ratio=0.5,
                       apply_wchannel='TGn', file_postfix='all20MHz')    # this case has consistent sampling rates (20 MHz) and applies a specific channel to all signals

    import matplotlib.pyplot as plt
    classes_slice_count = {}
    # compute an histogram that shows the distribution of labels in the dataset
    for k in ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']:
        classes_slice_count[k] = 0
        for key_idx, content in myds.ds_info['examples_map'][k].items():
             classes_slice_count[k] += len(content['slices'])

    plt.bar(list(classes_slice_count.keys()), list(classes_slice_count.values()))
    plt.show()

    for _ in range(10):
        index = np.random.choice(list(myds.ds_info['ixs_maps']['train'].keys()))
        obs, lbl = myds[index]

