from mmcv.runner.hooks import HOOKS, Hook
import numpy as np
import os
import cv2
import random
from mmcv.runner.hooks import LoggerHook
from mmcv.runner.dist_utils import master_only
import shutil
from mmcv import Config, ConfigDict


def search_and_modify_cfg(cfg, key, value):
    if isinstance(cfg, list):
        for e in cfg:
            search_and_modify_cfg(e, key, value)
    elif isinstance(cfg, (Config, ConfigDict)):
        for k, v in cfg.items():
            if k == key:
                cfg[k] = value
            else:
                search_and_modify_cfg(v, key, value)


def dict_file_to_list(dict_file):
    with open(dict_file, 'r') as f:
        lines = f.readlines()

    dict_list = ""
    for line in lines:
        char = line.rstrip("\n")
        if char == "":
            char = " "
        dict_list += char
    return dict_list


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


def prepare_dataset(ikdata, save_dir, split_ratio, seed):
    dataset_dir = os.path.join(save_dir, 'dataset')
    imgs_dir = os.path.join(dataset_dir, 'images')
    print("Preparing dataset...")
    for dire in [dataset_dir, imgs_dir]:
        if not (os.path.isdir(dire)):
            os.mkdir(dire)
        else:
            # delete files already in these directories to avoid mistakes
            for filename in os.listdir(dire):
                file_path = os.path.join(dire, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    train_label = os.path.join(dataset_dir, 'train_label.txt')
    test_label = os.path.join(dataset_dir, 'test_label.txt')

    for file in [train_label, test_label]:
        with open(file, "w") as f:
            f.write('')
    images = ikdata['images']
    n = len(images)
    if seed:
        random.seed(0)
    train_idx = random.sample(range(n), int(n * split_ratio))
    word_id = 1
    for img_id, sample in enumerate(images):
        img = cv2.imread(sample['filename'])
        for annot in sample["annotations"]:
            if 'bbox' in annot:
                x, y, w, h = annot['bbox']
            elif "segmentation_poly" in annot:
                pts = annot["segmentation_poly"]
                if len(pts) > 0:
                    x, y, w, h = polygone_to_bbox_xywh(pts[0])
            else:
                x, y, w, h = 0, 0, sample['width'], sample['height']

            if w > 0 and h > 0:
                txt = annot["text"]
                if len(txt) > 0:
                    word_img = img[int(y):int(y) + int(h), int(x):int(x) + int(w)]
                    word_img_name = os.path.join(imgs_dir, 'word_' + str(word_id) + '.png')
                    cv2.imwrite(word_img_name, word_img)
                    str_to_write = word_img_name + "\t" + txt + '\n'
                    if img_id in train_idx:
                        file_to_write = train_label
                    else:
                        file_to_write = test_label
                    with open(file_to_write, 'a') as f:
                        f.write(str_to_write)

                    word_id += 1

    print("Dataset prepared!")


class UserStop(Exception):
    pass


def register_mmlab_modules():
    # Define custom hook to stop process when user uses stop button and to save last checkpoint
    @HOOKS.register_module(force=True)
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
                runner.save_checkpoint(self.output_folder, "latest.pth", create_symlink=False)
                raise UserStop

    @HOOKS.register_module(force=True)
    class CustomMlflowLoggerHook(LoggerHook):
        """Class to log metrics and (optionally) a trained model to MLflow.
        It requires `MLflow`_ to be installed.
        Args:
            interval (int): Logging interval (every k iterations). Default: 10.
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`. Default: True.
            reset_flag (bool): Whether to clear the output buffer after logging.
                Default: False.
            by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        .. _MLflow:
            https://www.mlflow.org/docs/latest/index.html
        """

        def __init__(self,
                     log_metrics,
                     interval=10,
                     ignore_last=True,
                     reset_flag=False,
                     by_epoch=False):
            super(CustomMlflowLoggerHook, self).__init__(interval, ignore_last,
                                                         reset_flag, by_epoch)
            self.log_metrics = log_metrics

        @master_only
        def log(self, runner):
            tags = self.get_loggable_tags(runner)
            if tags:
                self.log_metrics(tags, step=self.get_iter(runner))
