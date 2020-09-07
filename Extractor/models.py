from django.db import models
from Extractor.utils import filename
from Extractor.tasks import inference_by_path
import os
from RetrievalExtractor import settings

from datetime import timedelta

class QueryModel(models.Model):
    image = models.ImageField(upload_to=filename.uploaded_date)
    feature = models.FileField(null=True)
    extract_time = models.DurationField(default=timedelta(days=0, hours=0))
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True, )

    def save(self, *args, **kwargs):
        super(QueryModel, self).save(*args, **kwargs)
        save_path = f'{self.image.path}.pth'
        result=inference_by_path.delay(self.image.path, save_path).get()
        self.extract_time=timedelta(seconds=result['extract_time'])
        self.feature = os.path.relpath(f'{self.image.url}.pth', settings.MEDIA_URL)
        super().save()
        print(self.feature.path)