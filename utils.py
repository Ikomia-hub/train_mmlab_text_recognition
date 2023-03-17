import numpy as np
import os
import cv2
import random
import shutil
import copy
import json

from mmengine.registry import HOOKS
from mmengine import Config, ConfigDict
from mmengine.hooks import Hook
from mmengine.hooks import LoggerHook
from mmengine.dist import master_only
from typing import Optional, Sequence, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]


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

    train_label = os.path.join(dataset_dir, 'instances_train.json')
    test_label = os.path.join(dataset_dir, 'instances_test.json')

    for file in [train_label, test_label]:
        with open(file, "w") as f:
            f.write('')
    images = ikdata['images']
    n = len(images)
    if seed:
        random.seed(0)
    train_idx = random.sample(range(n), int(n * split_ratio))

    json_train = \
        {
            "metainfo":
                {
                    "dataset_type": "TextRecogDataset",
                    "task_name": "textrecog",
                },
            'data_list': []
        }
    json_test = copy.deepcopy(json_train)

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
                    dict_to_write = {"img_path": word_img_name, "instances": [{"text": txt}]}
                    if img_id in train_idx:
                        json_train["data_list"].append(dict_to_write)
                    else:
                        json_test["data_list"].append(dict_to_write)

                    word_id += 1
    for json_file, json_dict in [(train_label, json_train), (test_label, json_test)]:
        with open(json_file, 'w') as f:
            f.write(json.dumps(json_dict))
    print("Dataset prepared!")


class UserStop(Exception):
    pass


def register_mmlab_modules():
    # Define custom hook to stop process when user uses stop button and to save last checkpoint
    @HOOKS.register_module(force=True)
    class CustomHook(Hook):
        # Check at each iter if the training must be stopped
        def __init__(self, stop, output_folder, emit_step_progress):
            self.stop = stop
            self.output_folder = output_folder
            self.emit_step_progress = emit_step_progress

        def after_epoch(self, runner):
            self.emit_step_progress()

        def _after_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Union[Sequence, dict]] = None,
                        mode: str = 'train') -> None:
            # Check if training must be stopped and save last model
            if self.stop():
                runner.save_checkpoint(self.output_folder, "latest.pth")
                raise UserStop

    @HOOKS.register_module(force=True)
    class CustomLoggerHook(LoggerHook):
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
                     interval=10):
            super(CustomLoggerHook, self).__init__(interval=interval, log_metric_by_epoch=True)
            self.log_metrics = log_metrics

        def after_val_epoch(self,
                            runner,
                            metrics=None) -> None:
            """All subclasses should override this method, if they need any
            operations after each validation epoch.

            Args:
                runner (Runner): The runner of the validation process.
                metrics (Dict[str, float], optional): Evaluation results of all
                    metrics on validation dataset. The keys are the names of the
                    metrics, and the values are corresponding results.
            """
            tag, log_str = runner.log_processor.get_log_after_epoch(
                runner, len(runner.val_dataloader), 'val')
            runner.logger.info(log_str)
            if self.log_metric_by_epoch:
                # when `log_metric_by_epoch` is set to True, it's expected
                # that validation metric can be logged by epoch rather than
                # by iter. At the same time, scalars related to time should
                # still be logged by iter to avoid messy visualized result.
                # see details in PR #278.
                metric_tags = {k: v for k, v in tag.items() if 'time' not in k}
                runner.visualizer.add_scalars(
                    metric_tags, step=runner.epoch, file_path=self.json_log_path)
                self.log_metrics(tag, step=runner.epoch)
            else:
                runner.visualizer.add_scalars(
                    tag, step=runner.iter, file_path=self.json_log_path)
                self.log_metrics(tag, step=runner.iter + 1)

        def after_train_iter(self,
                             runner,
                             batch_idx: int,
                             data_batch=None,
                             outputs=None):
            """Record logs after training iteration.

            Args:
                runner (Runner): The runner of the training process.
                batch_idx (int): The index of the current batch in the train loop.
                data_batch (dict tuple or list, optional): Data from dataloader.
                outputs (dict, optional): Outputs from model.
            """
            # Print experiment name every n iterations.
            if self.every_n_train_iters(
                    runner, self.interval_exp_name) or (self.end_of_epoch(
                runner.train_dataloader, batch_idx)):
                exp_info = f'Exp name: {runner.experiment_name}'
                runner.logger.info(exp_info)
            if self.every_n_inner_iters(batch_idx, self.interval):
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
                  and not self.ignore_last):
                # `runner.max_iters` may not be divisible by `self.interval`. if
                # `self.ignore_last==True`, the log of remaining iterations will
                # be recorded (Epoch [4][1000/1007], the logs of 998-1007
                # iterations will be recorded).
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            else:
                return
            runner.logger.info(log_str)
            runner.visualizer.add_scalars(
                tag, step=runner.iter + 1, file_path=self.json_log_path)
            self.log_metrics(tag, step=runner.iter + 1)