#!/usr/bin/env python
# coding: utf-8

# # Web Demo

# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import gc
from keras import backend as K

import torch.nn as nn
import numpy as np

import torchvision
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
from torchvision import transforms, datasets
from torch.autograd import Variable
import cv2
from PIL import Image
from copy import deepcopy
import os
from scipy.spatial.distance import cdist
import scipy.spatial.distance as distance
from torchvision.transforms import ToPILImage
from IPython.display import Image
from IPython.display import display, Image
from PIL import Image as Image
from math import*


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize_bk as visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = "./1817.jpg"


# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()


# In[4]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# In[5]:


class_names = ['person']


# ## Person segmentation

# In[6]:


#query image directory
IMAGE_DIR = "/data/Boyoung/dataset/New-dataset-train-test/train/1243/1106.jpg"
image = skimage.io.imread(IMAGE_DIR)

# Run detection
results = model.detect([image], verbose=1)

# save results
r = results[0]
query_result = visualize.display_instances(IMAGE_DIR, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


# In[12]:

from numba import cuda
cuda.select_device(0)
cuda.close()

# In[ ]:


import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


# ## model

# In[ ]:


class backbonequeryNet(nn.Module):
    def __init__(self):
        #query
        super(backbonequeryNet,self).__init__()
        self.backbone_query = torchvision.models.resnet50(pretrained=False)
        state_dict_query = self.backbone_query.state_dict()
        
        num_features_query = self.backbone_query.fc.in_features
        
        self.backbone_query = nn.Sequential(*list(self.backbone_query.children())[:-2])
        model_dict_query = self.backbone_query.state_dict()
        model_dict_query.update({k: v for k, v in state_dict_query.items() if k in model_dict_query})
        self.backbone_query.load_state_dict(model_dict_query)
        
        #query
        self.maxpooling_query = nn.MaxPool2d(7, stride=1)
        self.fc_query = nn.Linear(num_features_query, 1000)    
        
    def forward(self, x):
        x = self.backbone_query(x)
        maxpool_query = self.maxpooling_query(x)
        out_temp_query = self.fc_query(maxpool_query.view(maxpool_query.size(0), -1))
        out_query = F.normalize(out_temp_query, p=2, dim=1) #L2 normalized
            
        return out_query     

class backbonerelevantNet(nn.Module):
    def __init__(self):
        #non-relevant, relevant
        super(backbonerelevantNet,self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        state_dict = self.backbone.state_dict()
        
        num_features = self.backbone.fc.in_features
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        model_dict = self.backbone.state_dict()
        model_dict.update({k: v for k, v in state_dict.items() if k in model_dict})
        self.backbone.load_state_dict(model_dict)       
       
        #non-relevant, relevant
        self.maxpooling = nn.MaxPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, 1000)
        
    def forward(self,x):
        x = self.backbone(x)
        maxpool = self.maxpooling(x)
        out_temp = self.fc(maxpool.view(maxpool.size(0), -1))
        out = F.normalize(out_temp, p=2, dim=1) #L2 normalized
            
        return out 
    
class SiameseNet(nn.Module):
    def __init__(self,backbonequeryNet,backbonerelevantNet):
        super(SiameseNet, self).__init__()
        self.backbonequeryNet = backbonequeryNet
        self.backbonerelevantNet = backbonerelevantNet

    def forward(self, x1, x2, x3):
        output1 = self.backbonequeryNet(x1)
        output2 = self.backbonerelevantNet(x2)
        output3 = self.backbonerelevantNet(x3)
        
        return output1, output2, output3    


# In[ ]:


class TripletGalleryFashionDataset():
    # download, read data 하는 부분
    def __init__(self):
        self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
        
        self.gallery_data = ImageFolder(root='/data/Boyoung/dataset/Newdataset-maskrcnn/test/')
        self.gallery_label = self.gallery_data.imgs
    
    # 인덱스에 해당 아이템을 넘겨준다
    def __getitem__(self, index):
        
        img1= self.gallery_data[index][0]
        img_path = self.gallery_label[index][0]
        
        if self.transform is not None:
            img1 = self.transform(img1)
        
        
        return img1, img_path

    #data size를 넘겨주는 파트
    def __len__(self):
        return len(self.gallery_data)


# In[ ]:


gallery = ImageFolder(root='/data/Boyoung/dataset/Newdataset-maskrcnn/test/')


# In[ ]:


triplet_gallery_dataset = TripletGalleryFashionDataset()
triplet_gallery_loader = torch.utils.data.DataLoader(triplet_gallery_dataset, batch_size=1, shuffle=False)

data_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])


# In[ ]:


backbone_queryNet = backbonequeryNet()
backbone_relevantNet = backbonerelevantNet()

net = SiameseNet(backbone_queryNet, backbone_relevantNet)
net.cuda(0)


net.load_state_dict(torch.load('/data/Boyoung/dataset/New-dataset-model-maskrcnn/TNet_last_280.pth'))
feats = np.load('/data/Boyoung/dataset/new-dataset-maskrcnn-feature-train/all_feat_280_test.npy')

query_input = data_transform(transforms.ToPILImage()(query_result))
query_input = query_input.unsqueeze(1)
zero_arr1 = Variable(torch.zeros([1, 3, 224, 224]).cuda(0))
zero_arr2 = Variable(torch.zeros([1, 3, 224, 224]).cuda(0))

print(query_input.shape)
query_output, temp1, temp2 = net(Variable(query_input.cuda(0)), zero_arr1, zero_arr2)
query_feature = query_output.cpu().data.numpy()
dist = []
dist = cdist(query_feature, feats, 'cosine')[0]

ret = np.argsort(dist)[:30]

print("ret: ", ret)


for j in ret:
    path = gallery.imgs[j][0]
    print("path: ", path)
    im = Image.open(path)
    display(im)
    im = im.resize((256,256), Image.ANTIALIAS)
        


# In[ ]:




