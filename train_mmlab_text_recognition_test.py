import logging
from ikomia.core import task, ParamMap
from ikomia.utils.tests import run_for_test
import os
import ikomia
import yaml

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::train mmlab text recognition =====")
    input_dataset = t.get_input(0)
    params = task.get_parameters(t)
    plugins_folder = ikomia.ik_registry.getPluginsDirectory()
    plugin_folder = os.path.join(plugins_folder, "Python", t.name)
    configs_path = os.path.join(plugin_folder, "configs", "textrecog")
    input_dataset.load(data_dict["datasets"]["text"]["dataset_wildreceipt"])
    # loop on every configs available
    for directory in os.listdir(configs_path):
        if os.path.isdir(os.path.join(configs_path, directory)) and directory != "_base_":
            yaml_file = os.path.join(configs_path, directory, "metafile.yml")
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']
                model_dict = models_list[0]
                cfg = os.path.basename(model_dict["Config"])
                ckpt = model_dict["Weights"]
                params["cfg"] = cfg
                params["model_weight_file"] = ckpt
                params["model_name"] = directory
                params["epochs"] = 2
                params["batch_size"] = 1
                params["eval_period"] = 1

                task.set_parameters(t, params)
                yield run_for_test(t)
