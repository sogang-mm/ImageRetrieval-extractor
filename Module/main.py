from torchvision.transforms import transforms as trn
from PIL import Image
import torch
import os
from Module.dummy.model import Dummy
from datetime import datetime,timedelta

class Extractor:
    def __init__(self):
        # TODO
        #   - initialize and load model here
        code_path = os.path.dirname(os.path.abspath(__file__))
        self.device_id = int(os.environ['NVIDIA_USED_DEVICE_ID'])
        self.result = None
        self.model = Dummy().to(self.device_id)
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
        input_img = transform(img)
        start=datetime.now()
        feat = self.model(input_img.unsqueeze(0).to(self.device_id)).cpu()
        extract_time = datetime.now() - start
        torch.save(feat, save_to)
        result = {'extractor': self.model.__class__.__name__,
                  'image': image_path,
                  'save': save_to,
                  'extract_time': extract_time.total_seconds()
                  }

        return result
