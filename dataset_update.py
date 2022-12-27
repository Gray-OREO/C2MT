import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import h5py
import math
import yaml
from models.CertaintyCoeff import mini_CertaintyCoeff
import time


def default_loader(path):
    return Image.open(path).convert('RGB')


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


def NonOverlappingCropPatches(im, patch_size=32, stride=32):
    w, h = im.size
    patches = ()
    # local_patch = ()
    # patch_1 = torch.ones(3, 32, 32)
    # net = NONLocalBlock2D(3, sub_sample=True, bn_layer=True)
    for i in range(2, h - stride, stride):
        for j in range(2, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patch[0] = LocalNormalization(patch[0].numpy())
            patch[1] = LocalNormalization(patch[1].numpy())
            patch[2] = LocalNormalization(patch[2].numpy())
            # local_patch = net(patch.unsqueeze(0))
            # patch = torch.add(patch_1, local_patch.squeeze(0))
            patches = patches + (patch,)
    return patches


def stride_cal(patch_size, w):
    w_num = w // patch_size + 2
    w_stride = (w_num * patch_size - w) / w_num
    w_strides = np.ones([w_num - 1])
    w_strides[math.ceil((w_num - 1) // 2)] = 2
    w_strides = w_strides * w_stride

    # print(w_strides)
    return w_strides


def OverlappingCropPatches(im, patch_size=32):
    w, h = im.size
    patches = ()
    w_strides = stride_cal(patch_size, w)
    h_strides = stride_cal(patch_size, h)
    stride_w = 0
    stride_h = 0
    for i in range(w_strides.size + 1):
        for j in range(h_strides.size + 1):
            m = j * patch_size - stride_h
            n = i * patch_size - stride_w
            patch = to_tensor(im.crop((m, n, m + patch_size, n + patch_size)))
            if j < h_strides.size:
                stride_h += h_strides[j]
            else:
                stride_h = stride_h
            patch[0] = LocalNormalization(patch[0].numpy())
            patch[1] = LocalNormalization(patch[1].numpy())
            patch[2] = LocalNormalization(patch[2].numpy())
            # local_patch = net(patch.unsqueeze(0))
            # patch = torch.add(patch_1, local_patch.squeeze(0))
            patches = patches + (patch,)
        if i < w_strides.size:
            stride_w += w_strides[i]
        else:
            stride_w = stride_w
        stride_h = 0
    # print('my patch shape:', patches[0].shape)
    return patches


class CIQADataset(Dataset):
    def __init__(self, conf, exp_id=0, status='train', loader=default_loader):
        self.loader = loader
        im_dir = conf['im_dir']
        self.patch_size = conf['patch_size']
        # self.stride = conf['stride']
        datainfo = conf['datainfo']

        Info = h5py.File(datainfo, 'r')
        index = Info['index'][:, int(exp_id) % 1000]
        ref_ids = Info['ref_ids'][0, :]
        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))
        self.real_len = len(self.index)

        self.mos = Info['subjective_scores'][0, self.index]
        self.SRm = Info['distortion_types'][0, self.index]
        # image_name   im_names
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()
                    [::2].decode() for i in self.index]

        self.patches = ()
        self.label = []
        self.cls = []
        self.patch_index = []
        self.patch_num = 0
        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(im_names[idx]))
            im = self.loader(os.path.join(im_dir, im_names[idx]))
            patches = OverlappingCropPatches(im, self.patch_size)
            # patches = (torch.randn(3, 56, 56), torch.randn(3, 56, 56), torch.randn(3, 56, 56), torch.randn(3, 56, 56),
            #            torch.randn(3, 56, 56), torch.randn(3, 56, 56), torch.randn(3, 56, 56), torch.randn(3, 56, 56))
            self.patch_num = len(patches)
            if status == 'train':
                self.patches = self.patches + patches  #
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])
                    self.cls.append(self.SRm[idx])
                    self.patch_index.append(np.array([idx, i]))

            else:
                self.patches = self.patches + (torch.stack(patches),)  #
                self.label.append(self.mos[idx])
                self.cls.append(self.SRm[idx])
                self.patch_index.append(np.array([-1]))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], torch.Tensor([self.label[idx]]), torch.Tensor([self.cls[idx]]).long(), \
               self.patch_index[idx]


class new_CIQADataset(Dataset):
    def __init__(self, conf, p_label, exp_id=0, certainty=None, threshold=0.9, loader=default_loader):
        self.loader = loader
        im_dir = conf['im_dir']
        self.patch_size = conf['patch_size']
        # self.stride = conf['stride']
        datainfo = conf['datainfo']

        Info = h5py.File(datainfo, 'r')
        index = Info['index'][:, int(exp_id) % 1000]
        ref_ids = Info['ref_ids'][0, :]
        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)

        self.index = train_index

        self.mos = Info['subjective_scores'][0, self.index]
        self.SRm = Info['distortion_types'][0, self.index]
        # image_name   im_names
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()
                    [::2].decode() for i in self.index]

        self.patches = ()
        self.label = []
        self.cls = []
        self.patch_index = []
        self.patch_num = 0
        self.update_num = 0
        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(im_names[idx]))
            im = self.loader(os.path.join(im_dir, im_names[idx]))
            patches = OverlappingCropPatches(im, self.patch_size)
            # patches = (torch.randn(3, 56, 56), torch.randn(3, 56, 56), torch.randn(3, 56, 56), torch.randn(3, 56, 56),
            #            torch.randn(3, 56, 56), torch.randn(3, 56, 56), torch.randn(3, 56, 56), torch.randn(3, 56, 56))
            self.patch_num = len(patches)
            tmp = []
            for i in range(len(patches)):
                if certainty[idx, i] > threshold:
                    self.patches = self.patches + (patches[i],)
                    self.label.append(p_label[idx, i])
                    tmp.append(p_label[idx, i])
                    self.cls.append(self.SRm[idx])
                    self.patch_index.append(np.array([idx, i]))
                else:
                    self.patches = self.patches + (patches[i],)
                    self.label.append(self.mos[idx])
                    self.cls.append(self.SRm[idx])
                    self.patch_index.append(np.array([idx, i]))
            delta_mos = -(sum(tmp) - len(tmp) * self.mos[idx]) / (self.patch_num - len(tmp))
            self.update_num += len(tmp)
            for i in range(len(patches)):
                if certainty[idx, i] > threshold:
                    continue
                else:
                    self.label[i] = self.mos[idx] + delta_mos

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], (torch.Tensor([self.label[idx]])), torch.Tensor([self.cls[idx]]).long(), \
               self.patch_index[idx]


def get_new_train_loaders(config, p_label, certainty, train_batch_size, threshold=0.9, exp_id=0, num_workers=0):
    train_dataset = new_CIQADataset(config, p_label, exp_id, certainty, threshold)
    print('=========> Training dataset updated {} / {} samples!'.format(
        train_dataset.update_num, train_dataset.patch_num * len(train_dataset.index)))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    return train_loader


if __name__ == '__main__':
    with open('config.yaml', mode='r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(config['MA'])

    train_dataset = CIQADataset(config, exp_id=0, status='train')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=13, shuffle=True, num_workers=4)

    val_dataset = CIQADataset(config, exp_id=0, status='val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    test_dataset = CIQADataset(config, exp_id=0, status='test')
    test_loader = torch.utils.data.DataLoader(test_dataset)

    certainty_coeff = torch.zeros(train_dataset.real_len + len(val_dataset) + len(test_dataset),
                                  train_dataset.patch_num, device='cuda')
    pseudo_label = torch.zeros(train_dataset.real_len + len(val_dataset) + len(test_dataset), train_dataset.patch_num,
                               device='cuda')

    # learning framework
    for epoch in range(6):
        print('\nLearning epoch {} ...'.format(epoch))
        # train
        t0 = time.time()
        for i, (patches, label, cls, info) in enumerate(train_loader):
            # print(patches.shape)
            # print(label.shape)
            # print('--------')
            # print(type(samples[2]))
            # print(cls.shape)

            pred = torch.randn(info.shape[0], device='cuda')  # prediction
            t1 = time.time()
            mini_CC = mini_CertaintyCoeff(pred, label.to('cuda'))
            t2 = time.time()
            # print('mini_CCoeff_time:{:.4f}'.format(t2 - t1))
            for e in range(info.shape[0]):
                certainty_coeff[info[e, 0], info[e, 1]] = mini_CC[e]
                pseudo_label[info[e, 0], info[e, 1]] = pred[e]
            t3 = time.time()
            # print('CCoeff_time:{:.4f}'.format(t3 - t2))
            # print('Iter_time:{:.4f}'.format(t3 - t0))
            t0 = t3

        # print(certainty_coeff.shape)
        # print(certainty_coeff)

        if (epoch + 1) % 2 == 0:
            train_loader = get_new_train_loaders1(config, pseudo_label, certainty_coeff, train_batch_size=13,
                                                  threshold=0.95, exp_id=0, num_workers=1)
