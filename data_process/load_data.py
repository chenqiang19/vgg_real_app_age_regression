import os
import h5py
import torch
import torch.utils.data as data
import numpy as np


#load data(include origin images, real age and apprance age)
def load_data_set(data_dir, set_name):
    with h5py.File(os.path.join(data_dir, set_name, set_name+'_samples.h5'), 'r') as hf:
        data_set = hf[set_name+'_samples'][:]

    with h5py.File(os.path.join(data_dir, set_name, set_name+'_real_labels.h5'), 'r') as hf:
        data_real_labels = hf[set_name+'_real_labels'][:]

    with h5py.File(os.path.join(data_dir, set_name, set_name+'_app_labels.h5'), 'r') as hf:
        data_app_labels = hf[set_name+'_app_labels'][:]

    return data_set, data_real_labels, data_app_labels


#load extra feature(include gendre, race, happiness and makeup)
def load_data_extra_set(data_dir, set_name):
    with h5py.File(os.path.join(data_dir, set_name, set_name+'_all_extra_labels.h5'), 'r') as hf:
        all_extra_labels = hf[set_name+'_all_extra_labels'][:]

    return all_extra_labels


class dataSet(data.Dataset):
    def __init__(self, data_set, real_labels, app_labels, all_extra_attr, transform=None, target_transform=None):
        self.data_sets = data_set
        self.real_labels = real_labels
        self.app_labels = app_labels
        self.all_extra_attr = all_extra_attr
        self.transform = transform
        self.target_transform = target_transform

        assert self.data_sets.shape[0] == self.real_labels.shape[0] == self.app_labels.shape[0] == \
               self.all_extra_attr.shape[0], "data dim is not equal, please check again"

        self.data_sets = np.transpose(self.data_sets, (0, 3, 1, 2))
        self.data = []

        for i in range(self.data_sets.shape[0]):
            self.data.append((self.data_sets[i], self.real_labels[i], self.app_labels[i], self.all_extra_attr[i]))

    def __getitem__(self, index):
        data_set, real_label, app_label, all_extra_attr = self.data[index]

        if self.transform is not None:
            data_set = self.transform(data_set)

        if self.target_transform is not None:
            real_label = self.target_transform(real_label)
            app_label = self.target_transform(app_label)
            all_extra_attr = self.target_transform(all_extra_attr)

        if isinstance(data_set, np.ndarray):
            data_set = torch.from_numpy(data_set)
        # if isinstance(real_label, np.float64):
        #     real_label = torch.LongTensor(float(real_label))
        # if isinstance(app_label, np.float64):
        #     app_label = torch.LongTensor(float(app_label))
        if isinstance(all_extra_attr, np.ndarray):
            all_extra_attr = torch.from_numpy(all_extra_attr)

        return data_set, real_label, app_label, all_extra_attr

    def __len__(self):
        return len(self.data)
