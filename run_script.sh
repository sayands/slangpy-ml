# Training 
xvfb-run python main.py --mode train --max_epochs 10000 --save_path my_model.npz 
# Inference 
xvfb-run python main.py --mode inference --save_path my_model.npz --resolution 512