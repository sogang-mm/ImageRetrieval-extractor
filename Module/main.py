from torchvision.transforms import transforms as trn
from PIL import Image
import torch
import os
from Module.siamese.backbonequeryNet import backbonequeryNet
from Module.siamese.backbonerelevantNet import backbonerelevantNet
from Module.siamese.SiameseNet import SiameseNet
from datetime import datetime,timedelta

class Extractor:
    def __init__(self):
        # TODO
        #   - initialize and load model here
        code_path = os.path.dirname(os.path.abspath(__file__))
        self.device_id = int(os.environ['NVIDIA_USED_DEVICE_ID'])
        self.result = None

        self.model = SiameseNet(backbonequeryNet(), backbonerelevantNet()).to(self.device_id)
        ckpt = torch.load('/workspace/Module/TNet_last_282.pth')
        self.model.load_state_dict(ckpt)
        self.model.eval()

    @torch.no_grad()
    def inference_by_path(self, image_path, save_to):
        # TODO
        #   - Inference using image path

        transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path).convert("RGB")
        input_img = transform(img).unsqueeze(0).to(self.device_id)
        dummy1 = torch.zeros([1, 3, 224, 224]).to(self.device_id)
        dummy2 = torch.zeros([1, 3, 224, 224]).to(self.device_id)
        start = datetime.now()
        feat, _, _ = self.model(input_img, dummy1, dummy2)
        extract_time = datetime.now() - start
        torch.save(feat.cpu(), save_to)
        result = {'extractor': self.model.__class__.__name__,
                  'image': image_path,
                  'save': save_to,
                  'extract_time': extract_time.total_seconds()
                  }

        return result
