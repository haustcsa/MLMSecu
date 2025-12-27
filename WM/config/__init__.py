
import yaml
from easydict import EasyDict

with open("E:/scpcode/WM/config/train.yaml", "r") as f:
    training_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
