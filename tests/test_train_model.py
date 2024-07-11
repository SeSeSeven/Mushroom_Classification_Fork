import pytest
from pathlib import Path
from tests import _PROJECT_ROOT, _PATH_DATA
#import shutil
import os
from PIL import Image
from mushroom_classification.models.train_model import train_model
import torch
from omegaconf import OmegaConf

@pytest.fixture(scope="module")
def setup_directories():
    processed_dir = Path(_PATH_DATA, 'processed_test')
    if not os.path.exists(processed_dir):
        for split in ["train", "validation"]:
            if split == "train":
                im_num = 6
            else:
                im_num = 2
            for category in ["cat", "dog"]:
                category_path = Path(processed_dir, split, category)
                category_path.mkdir(parents=True, exist_ok=True)
                for i in range(im_num):
                    dummy_image = torch.rand(3, 300, 300) * 255
                    fake_im = Image.fromarray(dummy_image.byte().permute(1, 2, 0).numpy())
                    fake_im.save(Path(category_path, f'dummy_{i}.png'))
                    
    model_dir = Path(_PROJECT_ROOT, 'models_test')
    model_dir.mkdir(parents=True, exist_ok=True)

    yield processed_dir, model_dir

    # Clean up after tests
    # shutil.rmtree(processed_dir, ignore_errors=True)
    # shutil.rmtree(model_dir, ignore_errors=True)

@pytest.mark.run(order=4)
def test_train_model(setup_directories):
    processed_dir, model_dir= setup_directories

    config_dict = {
        'data_dir': processed_dir,
        'sweep': False,
        'model': {
            'model_name': 'resnet50.a1_in1k',
            'num_classes': 2,
            'data_transforms': {
                'train': [
                    {'_target_': 'torchvision.transforms.Resize', 'size': 235, 
                     "interpolation": 3, "max_size": None, "antialias": True},
                    {"_target_": "torchvision.transforms.CenterCrop", "size": 224},
                    {'_target_': 'torchvision.transforms.ToTensor'},
                    {'_target_': 'torchvision.transforms.Normalize', 
                     'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
                ],
                'valid': [
                    {'_target_': 'torchvision.transforms.Resize', 'size': 235, 
                     "interpolation": 3, "max_size": None, "antialias": True},
                    {"_target_": "torchvision.transforms.CenterCrop", "size": 224},
                    {'_target_': 'torchvision.transforms.ToTensor'},
                    {'_target_': 'torchvision.transforms.Normalize', 
                     'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
                ]
            },
            'save_path': model_dir,
            'save_name': 'test_model',
        },
        'hyperparameters': {
            'batch_size': 2,
            'learning_rate': 0.001,
            'epochs': 1
        }
    }

    cfg = OmegaConf.create(config_dict)
    train_model(cfg)

    # Check if model checkpoint is saved
    assert os.path.exists(Path(model_dir, 'test_model.ckpt')), "Model checkpoint ckpt not saved"
    assert os.path.exists(Path(model_dir, 'test_model.pt')), "Model checkpoint pt not saved"

