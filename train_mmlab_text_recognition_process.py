# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import utils, dataprocess
from ikomia.core.task import TaskParam
import os
from ikomia.dnn import datasetio, dnntrain
from ikomia.core import config as ikcfg
import copy
from datetime import datetime
from pathlib import Path
from train_mmlab_text_recognition.utils import prepare_dataset, UserStop, dict_file_to_list, register_mmlab_modules, \
    search_and_modify_cfg
import os.path as osp
import time
import mmcv
import torch
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmocr import __version__
from mmocr.apis import train_detector
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.utils import collect_env, get_root_logger
# importing pipelines enable registry
import mmocr.datasets.pipelines


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainMmlabTextRecognitionParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        self.cfg["model_name"] = "satrn"
        self.cfg["cfg"] = "satrn_small.py"
        self.cfg["weights"] = "https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_small_20211009-2cf13355.pth"
        self.cfg["custom_cfg"] = ""
        self.cfg["pretrain"] = True
        self.cfg["epochs"] = 10
        self.cfg["batch_size"] = 32
        self.cfg["dataset_split_ratio"] = 90
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"
        self.cfg["eval_period"] = 1
        self.cfg["dataset_folder"] = os.path.dirname(os.path.realpath(__file__))
        self.cfg["expert_mode"] = False
        self.cfg["seed"] = True

    def setParamMap(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["cfg"] = param_map["cfg"]
        self.cfg["custom_cfg"] = param_map["custom_cfg"]
        self.cfg["weights"] = param_map["weights"]
        self.cfg["pretrain"] = utils.strtobool(param_map["pretrain"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["dataset_split_ratio"] = int(param_map["dataset_split_ratio"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["eval_period"] = int(param_map["eval_period"])
        self.cfg["dataset_folder"] = param_map["dataset_folder"]
        self.cfg["expert_mode"] = utils.strtobool(param_map["expert_mode"])
        self.cfg["seed"] = utils.strtobool(param_map["seed"])


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainMmlabTextRecognition(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        register_mmlab_modules()

        # Variable to check if the training must be stopped by user
        self.stop_train = False

        self.max_seq_len = 26

        self.output_folder = ""

        # Create parameters class
        if param is None:
            self.setParam(TrainMmlabTextRecognitionParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        self.stop_train = False

        # Get param
        param = self.getParam()

        # Get input dataset
        input = self.getInput(0)

        # Current datetime is used as folder name
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        if len(input.data) == 0:
            print("ERROR, there is no input dataset")
            return

        # Output directory
        self.output_folder = Path(param.cfg["output_folder"] + "/" + str_datetime)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Tensorboard
        tb_logdir = os.path.join(ikcfg.main_cfg["tensorboard"]["log_uri"], str_datetime)

        # Transform Ikomia dataset to ICDAR compatible dataset if needed
        prepare_dataset(input.data, param.cfg["dataset_folder"], param.cfg["dataset_split_ratio"] / 100,
                        param.cfg["seed"])

        # Create config from config file and parameters
        if not (param.cfg["expert_mode"]):
            config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "textrecog",
                                  param.cfg["model_name"], param.cfg["cfg"])
            cfg = Config.fromfile(config)
            cfg.work_dir = str(self.output_folder)
            eval_period = param.cfg["eval_period"]
            cfg.load_from = param.cfg["weights"] if param.cfg["pretrain"] else None
            cfg.log_config = dict(
                interval=5,

                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type='TensorboardLoggerHook', log_dir=tb_logdir)
                ])
            cfg.total_epochs = param.cfg["epochs"]
            cfg.evaluation = dict(interval=eval_period, metric="acc", save_best="0_word_acc_ignore_case_symbol",
                                  rule="greater")
            cfg.dataset_type = 'OCRDataset'
            cfg.data_root = str(Path(param.cfg["dataset_folder"] + "/dataset"))
            cfg.data.train.datasets = [dict(
                type=cfg.dataset_type,
                img_prefix="",
                ann_file=cfg.data_root + '/train_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='\t')),
                pipeline=None,
                test_mode=False)]
            cfg.data.val.datasets = [dict(
                type=cfg.dataset_type,
                img_prefix="",
                ann_file=cfg.data_root + '/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='\t')),
                pipeline=None,
                test_mode=True)]
            cfg.data.samples_per_gpu = param.cfg["batch_size"]
            cfg.data.workers_per_gpu = 0
            cfg.data.val_dataloader = dict(samples_per_gpu=1)
            cfg.data.test_dataloader = dict(samples_per_gpu=1)

            cfg.log_config = dict(
                interval=5,

                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type='TensorboardLoggerHook', log_dir=tb_logdir)
                ])
            if "dict_file" in input.data["metadata"]:
                dict_list = dict_file_to_list(input.data["metadata"]['dict_file'])
            else:
                dict_list = list(tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                                       'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
                                       '*+,-./:;<=>?@[\\]_`~ '))
            # +2 for special tokens : <EOS> and <UNKNOWN>
            cfg.num_chars = len(dict_list) + 2
            cfg.label_convertor = dict(type=cfg.label_convertor.type,
                                       dict_type=None,
                                       dict_list=list(dict_list),
                                       with_unknown=True)
            cfg.model.label_convertor = cfg.label_convertor
            search_and_modify_cfg(cfg, "max_seq_len", self.max_seq_len)
            search_and_modify_cfg(cfg, "num_chars", cfg.num_chars)
            search_and_modify_cfg(cfg, "num_classes", cfg.num_chars)
            if "model" in cfg:
                if cfg.model is not None:
                    if "decoder" in cfg.model:
                        if cfg.model.decoder is not None:
                            search_and_modify_cfg(cfg.model.decoder, "pad_idx", cfg.num_chars - 1)

        else:
            config = param.cfg["custom_model"]
            cfg = Config.fromfile(config)

        gpus = 1
        launcher = "none"
        seed = None
        deterministic = True
        no_validate = cfg.evaluation.interval <= 0
        cfg.checkpoint_config = None

        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.gpu_ids = range(1) if gpus is None else range(gpus)
        # init distributed env first, since logger depends on the dist info.
        if launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(launcher, **cfg.dist_params)
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))

        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
        if seed is not None:
            logger.info(f'Set random seed to {seed}, '
                        f'deterministic: {deterministic}')
            set_random_seed(seed, deterministic=deterministic)
        cfg.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = osp.basename(config)

        datasets = [build_dataset(cfg.data.train)]

        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()

        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmocr_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES

        # add custom hook to stop process and save latest model each epoch
        cfg.custom_hooks = [
            dict(type='CustomHook', stop=self.get_stop, output_folder=str(self.output_folder),
                 emitStepProgress=self.emitStepProgress, priority='LOWEST'),
            dict(type='CustomMlflowLoggerHook', log_metrics=self.log_metrics)
        ]
        try:
            train_detector(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=(not no_validate),
                timestamp=timestamp,
                meta=meta)
        except UserStop:
            print("Training stopped by user")

        print("Training finished!")
        # Call endTaskRun to finalize process
        self.endTaskRun()

    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train = True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainMmlabTextRecognitionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_mmlab_text_recognition"
        self.info.shortDescription = "Training process for MMOCR from MMLAB in text recognition"
        self.info.description = "Training process for MMOCR from MMLAB in text recognition." \
                                "You can choose a predefined model configuration from MMLAB's " \
                                "model zoo or use custom models and custom pretrained weights " \
                                "by ticking Expert mode button."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Text"
        self.info.version = "1.0.1"
        self.info.iconPath = "icons/mmlab.png"
        self.info.authors = "Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin," \
                            " Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang," \
                            " Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua"
        self.info.article = "MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentationLink = "https://mmocr.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmocr"
        # Keywords used for search
        self.info.keywords = "train, mmlab, mmocr, ocr, text, recognition, pytorch, satrn, seg"

    def create(self, param=None):
        # Create process object
        return TrainMmlabTextRecognition(self.info.name, param)
