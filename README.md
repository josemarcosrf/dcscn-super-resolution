# DCSCN - Super Resolution

A pytorch implementation of "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network",
a deep learning based Single-Image Super-Resolution (SISR) model. [https://arxiv.org/abs/1707.05425](https://arxiv.org/abs/1707.05425)

## Project structure

```
├── checkpoints
├── data
│   ├── eval
│   │   ├── bsd100
│   │   ├── set14
│   │   └── set5
│   └── train
│       ├── bsd200
│       └── yang91
├── dcscn
│   ├── data_utils
│   │   ├── data_loader.py
│   ├── __init__.py
│   ├── net.py
│   └── trainer.py
├── logs
├── README.md
├── requirements.txt
├── setup.cfg
└── train.py

```

## Requirements

```
tqdm==4.28.1
matplotlib==2.2.3
numpy==1.13.0
scikit_image==0.13.1
Pillow==6.0.0
ai_utils==1.1.3
coloredlogs==10.0
skimage==0.0
torch==1.1.0
torchsummary==1.5.1
torchvision==0.2.2.post3
```
## How to

Basic training of a model (default configuration)
```bash
    python train.py
```

## TODO:

* Adapt **trainer** to perform appropiate metric comparison to save model
* Add typing all around the repo when appropiate
* Populate README
