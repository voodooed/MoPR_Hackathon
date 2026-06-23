from model.testing import segment_randlanet
from model.hyperparameters import hyp
from model.dataset import RandlanetDataset
from model.training import train_randlanet_model



segment_randlanet(model_path="data/saved_models/MoPR_whole/",
                  pc_path="data/pc_id=2/",
                  cfg=hyp,
                  num_workers=4, 
                  segmentation_name='example')

