# image-classification

## Installation

[https://pytorch.kr/get-started/locally/](https://pytorch.kr/get-started/locally/)

    ## 아나콘다 가상환경, 파이토치 설치.
    conda create --name cls-project python=3.9
    conda install pytorch torchvision torchaudio torchtext=0.18.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install pytorch-lightning==2.0.0 hydra-core==1.3.2 -c conda-forge

    pip install opencv-python albumentations augraphy tensorboardX timm transformers seaborn

    ## 주피터 노트북 커널
    pip install pexpect jupyter ipykernel
    pip uninstall pyzmq
    pip install pyzmq

설치가 정상적으로 되었는지 검사

```python
import torch

torch.__version__ ## '2.3.1'
torch.cuda.is_available() ## True
```

## Sample Data

[Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease/data)

    mkdir datasets
    cd datasets
    kaggle datasets download -d emmarex/plantdisease
    unzip plantdisease
    
    python reformat.py