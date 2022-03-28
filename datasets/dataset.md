
利用原始数据集生成cyclegan训练数据集，运行 
RSDD2CycleDataset.py

训练cyclegan
python train.py --dataroot ./datasets/RSDDs1_cycle --model cycle_gan --name RSDDs1_cycle --load_size 256 --crop_size 256
python train.py --dataroot ./datasets/RSDDs2_cycle --model cycle_gan --name RSDDs2_cycle --load_size 256 --crop_size 256
python train.py --dataroot ./datasets/KSDD_cycle --model cycle_gan --name KSDD_cycle --load_size 512 --crop_size 512
python train.py --dataroot ./datasets/DAGM_Class1_cycle/ --model cycle_gan --name DAGM_Class1_cycle --load_size 512 --crop_size 512 --n_epochs 25 --n_epochs_decay 25 --input_nc 1 --output_nc 1 --four_rotate --netG unet_256 --iffid --batch_size 4 --display_port 8097 --gpu_ids 0,1
python train.py --dataroot ./datasets/DAGM_Class2_cycle/ --model cycle_gan --name DAGM_Class2_cycle --load_size 512 --crop_size 512 --n_epochs 50 --n_epochs_decay 50 --input_nc 1 --output_nc 1 --four_rotate --netG unet_256 --iffid --batch_size 1 --gpu_ids 1 --display_port 8097
python train.py --dataroot ./datasets/DAGM_Class4_filted/ --model generate_defect --name DAGM_Class4_filted --load_size 512 --crop_size 512 --n_epochs 25 --n_epochs_decay 25 --input_nc 1 --output_nc 1 --four_rotate


利用cyclegan的R生成pix2pix的训练数据集，运行
python ./datasets/A_generate_B.py --datadir ./datasets/RSDDs1_cycle --modelpath ./checkpoints/RSDDs1_cycle/195_net_G_A.pth --load_size 256 --crop_size 256
python ./datasets/A_generate_B.py --datadir ./datasets/RSDDs2_cycle --modelpath ./checkpoints/RSDDs2_cycle/135_net_G_A.pth --load_size 256 --crop_size 256
python ./datasets/A_generate_B.py --datadir ./datasets/KSDD_cycle --modelpath ./checkpoints/KSDD_cycle/135_net_G_A.pth --load_size 512 --crop_size 512
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class1_cycle --modelpath ./checkpoints/DAGM_Class1_cycle_old/100_net_G_A.pth --load_size 512 --crop_size 512
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class2_cycle --modelpath ./checkpoints/DAGM_Class2_cycle_old/25_net_G_A.pth --load_size 512 --crop_size 512
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class3_cycle --modelpath ./checkpoints/DAGM_Class3_cycle_old/50_net_G_A.pth --load_size 512 --crop_size 512 --input_nc 1 --output_nc 1
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class4_cycle --modelpath ./checkpoints/DAGM_Class4_cycle/60_net_G_A.pth --load_size 512 --crop_size 512 --netG unet_256 --input_nc 1 --output_nc 1
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class5_cycle --modelpath ./checkpoints/DAGM_Class5_cycle_old/45_net_G_A.pth --load_size 512 --crop_size 512
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class6_cycle --modelpath ./checkpoints/DAGM_Class6_cycle_old/40_net_G_A.pth --load_size 512 --crop_size 512
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class7_cycle --modelpath ./checkpoints/DAGM_Class7_cycle_old/35_net_G_A.pth --load_size 512 --crop_size 512
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class8_cycle --modelpath ./checkpoints/DAGM_Class8_cycle_old/50_net_G_A.pth --load_size 512 --crop_size 512
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class9_cycle --modelpath ./checkpoints/DAGM_Class9_cycle_old/50_net_G_A.pth --load_size 512 --crop_size 512
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class10_cycle --modelpath ./checkpoints/DAGM_Class10_cycle_old/50_net_G_A.pth --load_size 512 --crop_size 512 --netG unet_256 --input_nc 1 --output_nc 1

然后运行
python ./datasets/combine_A_and_B.py --fold_A ./datasets/RSDDs1_cycle_A_and_B/A --fold_B ./datasets/RSDDs1_cycle_A_and_B/B --fold_AB ./datasets/RSDDs1_pix2pix_AB
python ./datasets/combine_A_and_B.py --fold_A ./datasets/RSDDs2_cycle_A_and_B/A --fold_B ./datasets/RSDDs2_cycle_A_and_B/B --fold_AB ./datasets/RSDDs2_pix2pix_AB
python ./datasets/combine_A_and_B.py --fold_A ./datasets/KSDD_cycle_A_and_B/A --fold_B ./datasets/KSDD_cycle_A_and_B/B --fold_AB ./datasets/KSDD_pix2pix_AB
python ./datasets/combine_A_and_B.py --fold_A ./datasets/DAGM_Class2_cycle_A_and_B/A --fold_B ./datasets/DAGM_Class2_cycle_A_and_B/B --fold_AB ./datasets/DAGM_Class2_pix2pix_AB

训练pix2pix
python train.py --dataroot ./datasets/RSDDs1_pix2pix_AB --model pix2pix --name RSDDs1_pix2pix --direction BtoA --load_size 256 --crop_size 256 --ifeval --print_freq 5000
python train.py --dataroot ./datasets/RSDDs2_pix2pix_AB --model pix2pix --name RSDDs2_pix2pix --direction BtoA --load_size 256 --crop_size 256
python train.py --dataroot ./datasets/KSDD_pix2pix_AB --model pix2pix --name KSDD_pix2pix --direction BtoA --load_size 512 --crop_size 512
python train.py --dataroot ./datasets/DAGM_Class8_pix2pix_AB --model pix2pix --name DAGM_Class8_pix2pix --direction BtoA --load_size 512 --crop_size 512 --ifeval --input_nc 1 --output_nc 1 --n_epochs 25 --n_epochs_decay 25 --display_port 8097 --gpu_ids 1

训练unet
python train_unet.py --dataset DAGM_Class1

运行
python eval.py --load_size 512 --epoch_count 50 --lr 0.01 --input_nc 1 --suffix _old --modelpath latest --epoch 100 --threshold 20 --eval_dataset_name DAGM_Class1
python eval.py --load_size 512 --epoch_count 50 --lr 0.01 --input_nc 1 --load_size 512 --crop_size 512 --output_nc 1 --modelpath 40 --epoch 50 --threshold3 127 --eval_dataset_name DAGM_Class10 --netG unet_256 --threshold 10 --threshold2 10 

--f --first_kernel --onlymax