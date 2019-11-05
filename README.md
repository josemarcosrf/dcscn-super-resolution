# DCSCN - Super Resolution

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fjmrf%2Fdcscn-super-resolution.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fjmrf%2Fdcscn-super-resolution?ref=badge_shield)

-------

A pytorch implementation of "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network",
a deep learning based Single-Image Super-Resolution (SISR) model. [https://arxiv.org/abs/1707.05425](https://arxiv.org/abs/1707.05425)

## Project structure

As output of `tree -L 3 -I "*.pyc|*cache*|*init*"`:
```
├── checkpoints
├── data
│   ├── eval            # evaluation data (no augmentation)
│   │   ├── bsd100
│   │   ├── set14
│   │   └── set5
│   └── train           # training dataset (no augmentation)
│       ├── bsd200
│       └── yang91
├── dcscn
│   ├── data_utils              # data loading and augmentation
│   │   ├── batcher.py
│   │   └── data_loader.py
│   ├── net.py                  # model definition
│   └── training                # training & helpers
│       ├── checkpointer.py
│       ├── metrics.py
│       ├── tf_logger.py
│       └── trainer.py
├── Dockerfile              # Dockerfile
├── entrypoint.sh           # entrypoint script
├── logs                    # tensorboard logs
├── README.md
├── requirements.txt
├── setup.cfg
├── tests                       # python unit tests
│   └── test_checkpointer.py
└── train.py                    # training entry point

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
torch==1.1.0
torchsummary==1.5.1
torchvision==0.2.2.post3
```
## How to

Basic training of a model (default configuration)
```bash
    python train.py
```

### Docker

#### Build
```bash
    docker build . -f Dockerfile -t dcscn
```

#### Run train
```bash
docker run -it \
    -v <project-root-dir>/checkpoints:/super-resolution/app/checkpoints \
    -v <project-root-dir>/data:/super-resolution/app/data \
    -v <project-root-dir>/logs:/super-resolution/app/logs \
    dcscn:latest run /bin/bash
```

## TODO:

* Add typing all around the repo when appropiate
* Verify training achieves paper described performance
* Generalise trainer into a better general purpose package
* Populate README


## License

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fjmrf%2Fdcscn-super-resolution.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fjmrf%2Fdcscn-super-resolution?ref=badge_large)
