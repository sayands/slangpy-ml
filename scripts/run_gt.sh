for mipmap_level in {0..14}
do
    mipmap_level_float=$(echo "scale=1; $mipmap_level / 2" | bc)
    echo $mipmap_level_float
    xvfb-run python run_gt.py --input /mnt/sdb/tejan/code/sayan_code/slangpy-ml/inputs/bernie.jpg --output /mnt/sdb/tejan/code/sayan_code/slangpy-ml/outputs/groundtruth/gt$mipmap_level_float.png --resolution 512 --mipmap_level=$mipmap_level_float
done