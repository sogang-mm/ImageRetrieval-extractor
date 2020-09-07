import numpy as np
import torch.nn as nn
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from torchvision.transforms import transforms as trn
from collections import OrderedDict
from model.nets import *
from dataset.loader import ListDataset


@torch.no_grad()
def extract(model, loader):
    features = []
    model.eval()
    bar = tqdm(loader, ncols=200, unit='batch')
    for i, (path, frame) in enumerate(loader):
        out = net(frame)
        features.append(out.cpu().numpy())
        bar.update()
    features = np.concatenate(features)
    bar.close()
    return features


if __name__ == '__main__':
    imgs=np.char.add('/nfs_shared/ms/ImageRetrievalDemo/dataset/',np.load('/nfs_shared/ms/ImageRetrievalDemo/dataset/images/fashion2-list.npy'))
    print(imgs)

    save_to = '/nfs_shared/ms/ImageRetrievalDemo/dataset/features/vcdb_core-mobilenet_avg_part3-0.pth'
    # name_save_to = '/nfs_shared/ms/ImageRetrievalDemo/dataset/images/mirflickr_list.npy'

    # net = Resnet50_RMAC().cuda()
    net = MobileNet_AVG().cuda()
    ckpt_path = '/nfs_shared/ms/ImageRetrievalDemo/extractor/mobilenet_avg_part3.pth'
    ckpt = torch.load(ckpt_path)
    net.load_state_dict(ckpt['model_state_dict'])
    print(net)

    net = nn.DataParallel(net)
    net.eval()
    batch = 256

    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader = DataLoader(ListDataset(imgs, transform=None), batch_size=batch, shuffle=False, num_workers=4)
    features = extract(net, loader)

    # np.save(name_save_to, np.char.replace(np.array(imgs),'/nfs_shared/ms/ImageRetrievalDemo/dataset/',''))
    np.save(save_to,features)


