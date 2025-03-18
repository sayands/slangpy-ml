import os.path as osp
import numpy as np
from PIL import Image
from slangpy.backend import DeviceType, TextureLoader

# Local application imports
from app import App

def calculate_psnr(groundtruth, pred):
    mse = np.mean((pred - groundtruth) ** 2)
    psnr = 10 * np.log10(1 / mse)
    return psnr

if __name__ == '__main__':
    
    resolution = 512
    base_dir = '/mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs'
    groundtruth_path = osp.join(base_dir, 'texture/output2.0.png')
    pred_path = osp.join(base_dir, 'groundtruth/texture/gt2.0.png')
    
    groundtruth = np.array(Image.open(groundtruth_path)).astype('float32') / 255.0
    pred = np.array(Image.open(pred_path)).astype('float32') / 255.0
    
    psnr = calculate_psnr(groundtruth, pred)
    
    print(f"PSNR: {psnr}")