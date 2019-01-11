# 3D pose estimation for Computer Vision course

Based on the work of Martinez et al. (https://github.com/una-dinosauria/3d-pose-baseline).

## Configs
Default config file is on `config/default.yaml`. Add any other yaml file in this folder to override default config.

## Install guide
- Download [SURREAL](https://github.com/gulvarol/surreal) and change the `surreal.data_path` config in the config folder.
- Download pretrained hourglass model [here](https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation)
 and edit the `hourglass.pretrained_path` config.

### Training
Execute `main_surreal.py`

## Acknowledgements
- Martinez et al. [https://github.com/una-dinosauria/3d-pose-baseline] 3D pose baseline
- Naman Jain and Sahil Shah [https://github.com/Naman-ntc/Pytorch-Human-Pose-Estimation]
- Varol et al. for the SURREAL dataset [https://github.com/gulvarol/surreal]
