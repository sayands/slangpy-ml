# Inference
for mipmap_level in {0..14}
do
    mipmap_level_float=$(echo "scale=1; $mipmap_level / 2" | bc)
    xvfb-run python main.py --mode inference --mipmap_level=$mipmap_level_float --output /mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs/texture/output$mipmap_level_float.png --resolution 512
done