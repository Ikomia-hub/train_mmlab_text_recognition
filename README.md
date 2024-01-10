<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_mmlab_text_recognition/main/icons/mmlab.png" alt="Algorithm icon">
  <h1 align="center">train_mmlab_text_recognition</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_mmlab_text_recognition">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_mmlab_text_recognition">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_mmlab_text_recognition/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_mmlab_text_recognition.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Training process for MMOCR from MMLAB in text recognition.You can choose a predefined model configuration from MMLAB's model zoo or use custom models and custom pretrained model_weight_file by ticking Expert mode button.


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

To try this code snippet, you can download and extract from [wildreceipt](https://download.openmmlab.com/mmocr/data/wildreceipt.tar).
Then make sure you fill the parameter **dataset_folder** correctly.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add dataset
dataset = wf.add_task(name="dataset_wildreceipt", auto_connect=False)

# Set dataset parameters
dataset.set_parameters({'dataset_folder': 'dataset/folder'})

# Add algorithm
algo = wf.add_task(name="train_mmlab_text_recognition", auto_connect=True)

# Set parameters
algo.set_parameters({"batch_size": 8})

# Run training
wf.run()
```

- **model_name** (str, default="satrn"): Model name. Set **model_name** and **cfg** to choose which model to train. See code snippet below to know what are the possibilities.
- **cfg** (str, default="satrn_shallow-small_5e_st_mj.py"): Config.
- **epochs** (int, default=10): number of complete passes through the training dataset.
- **batch_size** (int, default=4): number of samples processed before the model is updated.
- **dataset_split_ratio** (int, default=90): in percentage, divide the dataset into train and evaluation sets ]0, 100[.
- **output_folder** (str): path to where the model will be saved. Default folder is "runs/" in the algorithm directory.
- **eval_period** (int, default=1): interval between evaluations.
- **dataset_folder** (str): path to where the dataset compatible with mmlab is stored. Default folder is "/dataset" in the algorithm directory.
- **use_expert_mode** (bool, default=False): set to True only if you know how mmlab works. Then you can set all the parameters in the mmlab config system and it will override every other parameters above.
- **config_file** (str, default=""): path to the .py config file. Only for custom models.
- **model_weight_file** (str, default=""): path to the .pth weight file. Only for custom models.

*Note*: parameter key and value should be in **string format** when added to the dictionary.

MMLab framework offers multiple models. To ease the choice of couple (model_name/cfg), you can call the function *get_model_zoo()* to get a list of possible values.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add kie algorithm
kie = wf.add_task(name="infer_mmlab_kie", auto_connect=True)

# Get list of possible models (model_name, model_config)
print(kie.get_model_zoo())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).
