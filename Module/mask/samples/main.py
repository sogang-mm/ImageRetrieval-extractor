from Modules.Mask.Extractor.mrcnn import visualize_bk as visualize
from Modules.Mask.Extractor.mrcnn import model as modellib
from Modules.Mask.Extractor.mrcnn import utils
from Modules.Mask.Extractor.InferenceConfig import InferenceConfig
from Modules.Mask.Extractor.nets import backbonequeryNet, backbonerelevantNet, SiameseNet
import keras.backend.tensorflow_backend as KK
import tensorflow as tf

from torch.autograd import Variable
from torchvision import transforms
import torch

from PIL import Image
import skimage.io
import cv2

import os


class Extractor:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        # model_path = os.path.join(self.path, "model.txt")
        # self.model = open(model_path, "r")
        MODEL_DIR = os.path.join(self.path, 'logs')
        COCO_MODEL_PATH = os.path.join(self.path, 'mask_rcnn_coco.h5')
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)
            print('download..............')
        self._MaskRCNN_class_names = ['person']
        SIAMESENET_MODEL_PATH = os.path.join(self.path, 'TNet_last_300.pth')
        # torch.cuda.set_device(0)

        tf_conf = tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=False)
        # tf_conf.gpu_options.allow_growth=True
        KK.set_session(tf.Session(config=tf_conf))

        self.MASK_RCNN = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=InferenceConfig())
        self.MASK_RCNN.load_weights(COCO_MODEL_PATH, by_name=True)
        print("Load Mask-RCNN model : {} .... ok".format(COCO_MODEL_PATH))

        self.model = SiameseNet(backbonequeryNet(), backbonerelevantNet())
        self.model.load_state_dict(torch.load(SIAMESENET_MODEL_PATH))
        self.model.cuda()
        print("Load SiameseNet model : {} .... ok".format(SIAMESENET_MODEL_PATH))

        self.dataTransform = transforms.Compose([transforms.Resize((256, 256)),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                 ])
        self.dataTransform_check = transforms.Compose([transforms.Resize((224, 224))])

    def segmentation(self, image_path):
        image = skimage.io.imread(image_path)
        result = self.MASK_RCNN.detect([image], verbose=True)[0]

        seg = visualize.display_instances2(image_path, result['rois'], result['masks'], result['class_ids'],
                                           self._MaskRCNN_class_names, result['scores'])

        return seg

    def inference_by_path(self, image_path, options):
        result = []
        # TODO
        #   - Inference using image path
        seg = self.segmentation(image_path)

        query = Variable(self.dataTransform(Image.fromarray(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))).unsqueeze(0).cuda())
        zero_arr1 = Variable(torch.zeros([1, 3, 224, 224]).cuda())
        zero_arr2 = Variable(torch.zeros([1, 3, 224, 224]).cuda())

        out, tmp1, tmp2 = self.model(query, zero_arr1, zero_arr2)
        print(out.dtype)

        result = str(out.cpu().data.numpy().tolist())
        self.result = [{'Mask': result}]

        return self.result
