python train.py --work-dir  test_macular --gpu-ids 0 --local_rank 0 --load-from /path/deeplabv3_unet_s5-d16_256x256_40k_hrf_20201226_094047-3a1fdf85.pth configs/unet/deeplabv3_unet_s5-d16_256x256_1k_macular.py
python test.py --local_rank 0 --show-dir test_macular/test_show  configs/unet/deeplabv3_unet_s5-d16_256x256_1k_macular.py  test_macular/latest.pth
python test.py  configs/unet/deeplabv3_unet_s5-d16_256x256_1k_macular.py  test_macular/latest.pth --eval mDice
