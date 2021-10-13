from mmcv.runner.hooks import HOOKS, Hook
import numpy as np
import os
from pathlib import Path
import cv2
import random

def polygone_to_bbox_xywh(pts):
    """
    :param pts: list of coordinates with xs,ys respectively even,odd indexes
    :return: array of the bounding box xywh
    """
    x = np.min(pts[0::2])
    y = np.min(pts[1::2])
    w = np.max(pts[0::2]) - x
    h = np.max(pts[1::2]) - y
    return [x, y, w, h]


def prepare_dataset(ikdata, save_dir, split_ratio):
    dataset_dir = str(Path(save_dir+'/dataset'))
    imgs_dir = str(Path(dataset_dir+'/images'))
    print("Preparing dataset...")
    for dire in [dataset_dir,imgs_dir]:
        if not(os.path.isdir(dire)):
            os.mkdir(dire)
        else:
            print("Dataset already prepared!")
            return

    train_label = str(Path(dataset_dir+'/train_label.txt'))
    test_label = str(Path(dataset_dir+'/test_label.txt'))

    for file in [train_label,test_label]:
        with open(file, "w") as f:
            f.write('')
    images = ikdata['images']
    n = len(images)
    train_idx = random.sample(range(n), int(n * split_ratio))
    word_id = 1
    for img_id, sample in enumerate(images):
        img = cv2.imread(sample['filename'])
        for annot in sample["annotations"]:
            pts = annot["segmentation_poly"]
            if len(pts[0]) > 2 :
                # [0] because we suppose text images have no hole in it ie. 1 shape for 1 word
                x,y,w,h = polygone_to_bbox_xywh(pts[0])
                if w>0 and h>0:
                    txt = annot["text"]
                    word_img = img[int(y):int(y)+int(h),int(x):int(x)+int(w)]
                    word_img_name = str(Path(imgs_dir) / ('word_'+str(word_id)+'.png'))
                    cv2.imwrite(word_img_name,word_img)
                    str_to_write = word_img_name + " " + txt+'\n'
                    if img_id in train_idx:
                        file_to_write = train_label
                    else:
                        file_to_write = test_label
                    with open(file_to_write,'a') as f:
                        f.write(str_to_write)

                    word_id+=1

    print("Dataset prepared!")

textrecog_models = {
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'crnn/crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config': 'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt': 'sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt': 'nrtr/nrtr_r31_academic_20210406-954db95e.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by8_1by4_academic_20210406-ce16e7cc.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'satrn/satrn_academic_20210809-59c8c92d.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'satrn/satrn_small_20210811-2badf6fc.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt': 'seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config': 'tps/crnn_tps_academic_dataset.py',
                'ckpt': 'tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
            }
        }

class UserStop(Exception):
    pass

# Define custom hook to stop process when user uses stop button and to save last checkpoint
try:
    @HOOKS.register_module()
    class CustomHook(Hook):
        # Check at each iter if the training must be stopped
        def __init__(self, stop, output_folder, emitStepProgress):
            self.stop = stop
            self.output_folder = output_folder
            self.emitStepProgress = emitStepProgress

        def after_epoch(self, runner):
            self.emitStepProgress()

        def after_train_iter(self, runner):
            # Check if training must be stopped and save last model
            if self.stop():
                runner.save_checkpoint(self.output_folder,"latest.pth",create_symlink=False)
                raise UserStop
except:
    pass