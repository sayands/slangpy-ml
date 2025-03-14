# Inference
# for mipmap_level in {0..14}
# do
#     mipmap_level_float=$(echo "scale=1; $mipmap_level / 2" | bc)
#     xvfb-run python main.py --mode inference --mipmap_level=$mipmap_level_float --output /mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs/texture/output$mipmap_level_float.png --resolution 512
# done

# xvfb-run python main.py --mode inference --mipmap_level=4.5 --output /mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs/checkerboard/output4.5.png --resolution 512

# xvfb-run python main.py --mode inference --mipmap_level=6.5 --output /mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs/bernie/output6.5.png --resolution 512

mipmap_level_float=7.0
xvfb-run python main.py --mode inference --mipmap_level=$mipmap_level_float --output /mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs/texture/output$mipmap_level_float.png --resolution 512