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

# Run training
wf.run()
```

## :pencil: Set algorithm parameters

- **dataset_folder** (str, default=""): path to the dataset folder.

*Note*: parameter key and value should be in **string format** when added to the dictionary.

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).


## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
dataset = wf.add_task(name="train_mmlab_text_recognition", auto_connect=False)

# Set dataset parameters
dataset.set_parameters({'dataset_folder': 'dataset/folder'})

# Load dataset
wf.run()

# Look at the loaded data
print(dataset.get_output(0).data)
```
