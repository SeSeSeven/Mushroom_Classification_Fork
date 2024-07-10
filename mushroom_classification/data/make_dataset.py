import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from mushroom_classification import _PATH_DATA
import shutil
import typer
from typing_extensions import Annotated
app = typer.Typer()

@app.command()
def split_data(raw_dir: Annotated[str,typer.Option("--raw_dir",'-d')]=str(Path(_PATH_DATA,'raw')), 
               processed_dir: Annotated[str,typer.Option("--processed_dir",'-p')]=str(Path(_PATH_DATA,'processed')), 
               val_size: Annotated[float,typer.Option("--val_size",'-v')]=0.15, 
               test_size: Annotated[float,typer.Option("--test_size",'-t')]=0.15, 
               random_state: Annotated[int,typer.Option("--random_state",'-r')]=42) -> None:
    
    """Split data into train, validation, and test sets."""

    categories = [d.name for d in Path(raw_dir).glob('*') if d.is_dir()]
    for category in categories:
        images = list(Path(raw_dir, category).glob('*'))
        nontest_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)
        train_images, val_images = train_test_split(nontest_images, test_size=val_size / (1 - test_size), random_state=random_state)
        
        train_path = Path(processed_dir, 'train', category)
        val_path = Path(processed_dir, 'validation', category)
        test_path = Path(processed_dir, 'test', category)

        shutil.rmtree(train_path, ignore_errors=True)
        shutil.rmtree(val_path, ignore_errors=True)
        shutil.rmtree(test_path, ignore_errors=True)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
        for img in train_images:
            shutil.copy(img, Path(train_path, img.name))
        for img in val_images:
            shutil.copy(img, Path(val_path, img.name))
        for img in test_images:
            shutil.copy(img, Path(test_path, img.name))

if __name__ == "__main__":
    app()

