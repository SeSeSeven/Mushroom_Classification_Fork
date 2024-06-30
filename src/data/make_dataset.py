import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import argparse

def split_data(dataset_path, output_path, val_size=0.15, test_size=0.15, random_state=42):
    categories = [d.name for d in Path(dataset_path).glob('*') if d.is_dir()]
    for category in categories:
        images = list(Path(dataset_path, category).glob('*'))
        nontest_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)
        train_images, val_images = train_test_split(nontest_images, test_size=val_size / (1 - test_size), random_state=random_state)
        
        os.makedirs(Path(output_path, 'train', category), exist_ok=True)
        os.makedirs(Path(output_path, 'val', category), exist_ok=True)
        os.makedirs(Path(output_path, 'test', category), exist_ok=True)
        
        for img in train_images:
            shutil.copy(img, Path(output_path, 'train', category, img.name))
        for img in val_images:
            shutil.copy(img, Path(output_path, 'val', category, img.name))
        for img in test_images:
            shutil.copy(img, Path(output_path, 'test', category, img.name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Main function for splitting dataset.",
    )

    parser.add_argument(
        "dataset_path", 
        type=str, 
        help="Dataset path containing class-folders with corresponding image data."
    )

    parser.add_argument(
        "output_path", 
        type=str, 
        help="Output path for splitted data folders."
    )

    parser.add_argument(
        "--val_size",
        "-v",
        default=0.15,
        type=float,
        help="Validation set size",
    )

    parser.add_argument(
        "--test_size",
        "-t",
        default=0.15,
        type=float,
        help="Test set size",
    )

    parser.add_argument(
        "--random_state",
        "-r",
        default=42,
        type=int,
        help="Random state for data splitting.",
    )

    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_path = args.output_path
    val_size = args.val_size
    test_size = args.test_size
    random_state = args.random_state

    split_data(dataset_path, output_path, val_size, test_size, random_state)

