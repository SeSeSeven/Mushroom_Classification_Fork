#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate mlops

python data/make_dataset.py ../dataset/Mushrooms ../dataset -v 0.15 -t 0.15 -r 42
python train_model.py
echo "done"

conda deactivate
