# Neural Texture Compression with Slang

Modified from: https://github.com/shader-slang/slangpy/tree/main/experiments/neuralnetwork

## Scripts
Here are some scripts for training and inference with the neural texture generator.

```bash
python main.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level $mipmap_level
python main.py --mode inference --mipmap_level=$mipmap_level_float --output output_path/output$mipmap_level_float.png --resolution 512
```
This will save the relevant generated texture at `mipmap_level_float` to the `output_path`.

### Training Residual Model
```bash
python main_res.py --mode train --max_epochs 1000 --save_dir checkpoints/ --mipmap_level 0
```

This script assumes that the base model is trained and saved.

### Evaluation

```bash
python run_gt.py --input input_path --output output_path --resolution 512 --mipmap_level=$mipmap_level_float (to run groundtruth)
python eval_runtime_inference.py (to evaluate runtime)
python eval.py (to evaluate quality)
```