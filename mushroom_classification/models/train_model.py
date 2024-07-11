import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import UnidentifiedImageError, Image
from pytorch_lightning.loggers import WandbLogger
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
from mushroom_classification import _PATH_CONF
from mushroom_classification.models.model import get_model
import wandb
import hydra
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """Override the __getitem__ method to skip corrupted images."""
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except (UnidentifiedImageError, OSError):
            print(f"Skipping corrupted image at index {index}")
            new_index = (index + 1) % len(self)
            return self.__getitem__(new_index)

class SaveModelPTCallback(Callback):
    """Save also .pt file of the model at the end of training"""
    def __init__(self, save_file):
        super().__init__()
        self.save_file = save_file
    def on_train_end(self, trainer, pl_module):
        torch.save(pl_module.state_dict(), self.save_file)

@hydra.main(version_base=None, config_path=_PATH_CONF, config_name="hydra")
def train_model(cfg: DictConfig) -> None:
    """Train a model using the provided hyperparameters."""
    print(OmegaConf.to_yaml(cfg))

    # Extract hyperparameters from Hydra configuration
    data_dir = cfg.data_dir
    sweep = cfg.sweep
    model_name = cfg.model.model_name
    num_classes = cfg.model.num_classes
    data_transforms = cfg.model.data_transforms
    save_path = cfg.model.save_path
    save_name = cfg.model.save_name

    # Mapping interpolation integer values to actual PIL.Image constants
    interpolation_map = {
        0: Image.NEAREST,
        2: Image.BILINEAR,
        3: Image.BICUBIC,
        1: Image.LANCZOS
    }
    # Update interpolation values to use the actual PIL.Image constants
    for transform in data_transforms.train:
        if "_target_" in transform and "interpolation" in transform:
            transform.interpolation = interpolation_map[transform.interpolation]
    for transform in data_transforms.valid:
        if "_target_" in transform and "interpolation" in transform:
            transform.interpolation = interpolation_map[transform.interpolation]
    # Instantiate data transformations
    train_transforms = transforms.Compose([instantiate(t) for t in data_transforms.train])
    valid_transforms = transforms.Compose([instantiate(t) for t in data_transforms.valid])

    if sweep:
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
        train_dataset = SafeImageFolder(root=Path(data_dir, 'train'), transform=train_transforms)
        val_dataset = SafeImageFolder(root=Path(data_dir, 'validation'), transform=valid_transforms)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(model_name, num_classes, lr)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_path,
        filename=save_name,
        save_top_k=1,
        mode='min',
    )

    save_model_pt_callback = SaveModelPTCallback(save_file=Path(save_path, save_name+'.pt'))

    if sweep:
        # Initialize WandbLogger
        wandb_logger = WandbLogger(project=os.getenv("WANDB_PROJECT"))

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",  # Automatically select available devices
        callbacks=[checkpoint_callback, save_model_pt_callback],
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
    yaml_path = Path(_PATH_CONF,'hydra.yaml')
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    # Get the count value
    use_sweep = config['sweep']
    # If use_sweep is False, train the model using just one set of hyperparameters
    # Else, run the sweep agent for multiple hyperparameter configurations
    if use_sweep:
        for name in ['API_KEY', 'PROJECT', 'ENTITY']:
            var = 'WANDB_'+name
            if not os.getenv(var):
                raise ValueError(f"Please set the {name} environment variable.")
        wandb.login() #os.getenv("WANDB_API_KEY")
        with open(os.path.join(_PATH_CONF, 'sweep.yaml')) as file:
            sweep_config = yaml.safe_load(file)
        sweep_id = wandb.sweep(sweep=sweep_config, project=os.getenv("WANDB_PROJECT"))
        wandb.agent(sweep_id, function=train_model, count=sweep_config['count'], 
                    project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY"))
    else:
        train_model()
        