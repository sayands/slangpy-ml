# Training at different mip map levels
xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 0
xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 1
xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 2
xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 3
xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 4
xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 5
xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 6
xvfb-run python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 7

# xvfb-run python main.py --mode inference --save_path my_model.npz --resolution 512