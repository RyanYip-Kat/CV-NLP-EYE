#CUDA_VISIBLE_DEVICES=0 python train.py --save UNet_vessel_seg --batch_size 16 --cuda True --val_on_test True --in_channels 3 --N_epochs 50
#CUDA_VISIBLE_DEVICES=0 python test.py --save UNet_vessel_seg
#CUDA_VISIBLE_DEVICES=0 python train.py --save UNet_EX_seg --batch_size  32 --cuda True --in_channels 3 --N_epochs 50 --train_data_path_list ./prepare_dataset/data_path_list/IDRID_EX/train.txt --test_data_path_list ./prepare_dataset/data_path_list/IDRID_EX/val.txt --img_size 512


CUDA_VISIBLE_DEVICES=3 python train.py --save UNet_OD_seg --batch_size  64 --cuda True --in_channels 3 --N_epochs 50 --train_data_path_list ./prepare_dataset/data_path_list/IDRID_OD/train.txt --test_data_path_list ./prepare_dataset/data_path_list/IDRID_OD/val.txt --img_size 512 --val_ratio 0.2 --val_on_test  True --start_epoch 1
