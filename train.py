from model.testing import segment_randlanet
from model.hyperparameters import hyp
from model.dataset import RandlanetDataset
from model.training import train_randlanet_model


train_randlanet_model(
    train_set_list = [
        "data/pc_id=1/"
    ],
    test_set_list = ["data/pc_id=2/"],
    hyperpars = hyp,
    use_mlflow = False,
    num_workers = 8,
    model_name = "MoPR_whole"
)


