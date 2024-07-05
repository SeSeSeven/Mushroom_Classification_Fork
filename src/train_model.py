import logging
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import UnidentifiedImageError, Image
from pytorch_lightning.loggers import WandbLogger
from torch.profiler import profile, ProfilerActivity
import wandb
import hydra
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

#import sys
#sys.path.insert(0, '/content/drive/MyDrive/mushroom_classfication')

from models.model import get_model

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(MyImageFolder, self).__getitem__(index)
        except (UnidentifiedImageError, OSError):
            print(f"Skipping corrupted image at index {index}")
            new_index = (index + 1) % len(self)
            return self.__getitem__(new_index)

@hydra.main(version_base=None, config_path="hydra_conf", config_name="config")
def train_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Extract hyperparameters from Hydra configuration
    data_dir = cfg.data_dir
    sweep = cfg.sweep
    model_name = cfg.model.model_name
    num_classes = cfg.model.num_classes
    data_transforms = cfg.model.data_transforms
    save_path = cfg.model.save_path

    # Mapping interpolation integer values to actual PIL.Image constants
    interpolation_map = {
        0: Image.NEAREST,
        2: Image.BILINEAR,
        3: Image.BICUBIC,
        1: Image.LANCZOS
    }
    # Update interpolation values to use the actual PIL.Image constants
    for transform in cfg.model.data_transforms.train:
        if "_target_" in transform and "interpolation" in transform:
            transform.interpolation = interpolation_map[transform.interpolation]
    for transform in cfg.model.data_transforms.valid:
        if "_target_" in transform and "interpolation" in transform:
            transform.interpolation = interpolation_map[transform.interpolation]
    # Instantiate data transformations
    train_transforms = transforms.Compose([instantiate(t) for t in cfg.model.data_transforms.train])
    valid_transforms = transforms.Compose([instantiate(t) for t in cfg.model.data_transforms.valid])

    if sweep:
        # If wandb.config is available, override the hydra configuration
        wandb.init()
        config = wandb.config
        if 'train' in config:
            batch_size = config.train.batch_size
            epochs = config.train.epochs
        if 'model' in config:
            lr = config.model.lr
    else:
        batch_size = cfg.hyperparameters.batch_size
        lr = cfg.hyperparameters.learning_rate
        epochs = cfg.hyperparameters.epochs
        
    try:
        train_dataset = MyImageFolder(root=f"{data_dir}/train", transform=train_transforms)
        val_dataset = MyImageFolder(root=f"{data_dir}/val", transform=valid_transforms)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_model('hf_hub:timm/'+model_name, num_classes, lr)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_path,
        filename='{model_name}-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    if sweep:
        # Initialize WandbLogger
        wandb_logger = WandbLogger(project=model_name)

    # Set accelerator to GPU if available, else fallback to CPU
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices="auto",  # Automatically select available devices
        callbacks=[checkpoint_callback],
        logger=wandb_logger if sweep else None
    )

    # Profile training step
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        trainer.fit(model, train_loader, val_loader)
        prof.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Main function for splitting dataset.",
    )
    parser.add_argument(
        "--use_sweep",
        "-s",
        type=bool,
        default=False,
        help="Use sweep or just train once",
    )
    parser.add_argument(
        "--project_name",
        "-p",
        type=str,
        default="mushroom_classification",
        help="Project Name",
    )
    parser.add_argument(
        "--wandb_api",
        "-w",
        type=str,
        default="mushroom_classification",
        help="Project Name",
    )
    parser.add_argument(
        "--wandb_entity",
        "-t",
        type=str,
        default="entity",
        help="Entity",
    )
    args = parser.parse_args()
    # If sweep ID is not provided, train the model using just one set of hyperparameters
    # Else, run the sweep agent for multiple hyperparameter configurations
    project_name = args.project_name
    use_sweep = args.use_sweep
    api = args.wandb_api
    entity = args.wandb_entity
    if use_sweep:
        wandb.login(api)
        cur_file_path = os.path.abspath(__file__)
        print(f"Current file path: {cur_file_path}")
        # Get the directory containing the current file
        cur_dir = os.path.dirname(cur_file_path)
        with open(os.path.join(cur_dir,'sweep.yaml')) as file:
            sweep_config = yaml.safe_load(file)
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        wandb.agent(sweep_id, function=train_model, count=4, project=project_name, entity=entity)
    else:
        train_model()
        
