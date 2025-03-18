#!/bin/bash

# Training at different resolutions
for mipmap_level in {0..7}
do
    xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level $mipmap_level
done

# Inference
for mipmap_level in {0..14}
do
    mipmap_level_float=$(echo "scale=1; $mipmap_level / 2" | bc)
    xvfb-run python main.py --mode inference --mipmap_level=$mipmap_level_float --output /mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs/checkerboard/output$mipmap_level_float.png --resolution 512
done