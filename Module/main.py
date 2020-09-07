from Module.mask.mrcnn import visualize_bk as visualize
from Module.mask.mrcnn import model as modellib
from Module.mask.mrcnn import utils
from Module.mask.InferenceConfig import InferenceConfig
# from Module.mask.visualize import visualize as vis
from Module.mask.siamese.nets import backbonequeryNet, backbonerelevantNet, SiameseNet

from torchvision.transforms import transforms as trn
from PIL import Image
import torch
import os

import keras.backend.tensorflow_backend as KK
import tensorflow as tf
import numpy as np

import skimage.io
import cv2
from datetime import datetime,timedelta

class Extractor:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        code_path = os.path.dirname(os.path.abspath(__file__))
        self.device_id = int(os.environ['NVIDIA_USED_DEVICE_ID'])
        self.result = None

        tf_conf = tf.ConfigProto(device_count={'GPU': self.device_id}, log_device_placement=False)
        tf_conf.gpu_options.per_process_gpu_memory_fraction=0.3
        # tf_conf.gpu_options.allow_growth =True

        KK.set_session(tf.Session(config=tf_conf))

        MODEL_DIR = os.path.join(self.path, 'logs')
        COCO_MODEL_PATH = os.path.join(self.path, 'mask_rcnn_coco.h5')
        SIAMESENET_MODEL_PATH = os.path.join(self.path, 'TNet_last_300.pth')


        self.model = SiameseNet(backbonequeryNet(), backbonerelevantNet()).to(self.device_id)
        self.model.load_state_dict(torch.load(SIAMESENET_MODEL_PATH))
        self.model.eval()
        print("Load SiameseNet model : {} .... ok".format(SIAMESENET_MODEL_PATH))

        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)
            print('download..............')

        self._MaskRCNN_class_names = ['person']
        self.MASK_RCNN = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=InferenceConfig())
        self.MASK_RCNN.load_weights(COCO_MODEL_PATH, by_name=True)
        print("Load Mask-RCNN model : {} .... ok".format(COCO_MODEL_PATH))



        self.transform = trn.Compose([trn.Resize((256, 256)),
                                     trn.CenterCrop(224),
                                     trn.ToTensor(),
                                     trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

    def segmentation(self, image_path):
        image = skimage.io.imread(image_path)
        start = datetime.now()
        result = self.MASK_RCNN.detect([image], verbose=True)[0]
        seg_time = datetime.now() - start
        seg = visualize.display_instances2(image_path, result['rois'], result['masks'], result['class_ids'],
                                           self._MaskRCNN_class_names, result['scores'])

        return seg,seg_time

    @torch.no_grad()
    def inference_by_path(self, image_path, save_to):
        # TODO
        #   - Inference using image path

        seg,seg_time = self.segmentation(image_path)
        extract_time = seg_time

        input_img = self.transform(Image.fromarray(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device_id)
        dummy1 = torch.zeros([1, 3, 224, 224]).to(self.device_id)
        dummy2 = torch.zeros([1, 3, 224, 224]).to(self.device_id)
        start = datetime.now()
        feat, _, _ = self.model(input_img, dummy1, dummy2)
        extract_time += (datetime.now() - start)

        torch.save(feat.cpu(), save_to)
        result = {'extractor': self.model.__class__.__name__,
                  'image': image_path,
                  'save': save_to,
                  'extract_time': extract_time.total_seconds()
                  }

        return result



