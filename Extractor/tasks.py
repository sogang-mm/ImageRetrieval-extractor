from __future__ import print_function

from RetrievalExtractor.celerys import app
from celery.signals import worker_init, worker_process_init
from billiard import current_process
from datetime import timedelta

@worker_init.connect
def model_load_info(**__):
    print("====================")
    print("Worker RetrievalExtractor Initialize")
    print("====================")


@worker_process_init.connect
def module_load_init(**__):
    global extractor
    worker_index = current_process().index

    print("====================")
    print(" Worker Id: {0}".format(worker_index))
    print("====================")

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    # from Modules.resnet50_rmac.main import Dummy
    # analyzer=Dummy()
    from Module.main import Extractor
    extractor = Extractor()


@app.task
def inference_by_path(image_path, save_to):


    print('img : {}  save_to : {}'.format(image_path, save_to))
    result = extractor.inference_by_path(image_path, save_to)
    print("extract-time: {}".format(timedelta(seconds=result['extract_time'])))
    return result




