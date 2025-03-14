#!/bin/bash

# # Training at different resolutions
# for mipmap_level in {0..7}
# do
#     xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level $mipmap_level
# done

# Inference
# for mipmap_level in {0..14}
# do
#     mipmap_level_float=$(echo "scale=1; $mipmap_level / 2" | bc)
#     xvfb-run python main.py --mode inference --mipmap_level=$mipmap_level_float --output /mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs/checkerboard/output$mipmap_level_float.png --resolution 512
# done


# # Training at different mip map levels
# xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 0
# xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 1
# xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 2
# xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 3
# xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 4
# xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 5
# xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 6
# xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 7

# # Inference
# xvfb-run python main.py --mode inference --mipmap_level=0 --output /mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs/output0.png --resolution 512