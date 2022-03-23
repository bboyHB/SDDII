
利用原始数据集生成cyclegan训练数据集，运行 
RSDD2CycleDataset.py

训练cyclegan
python train.py --dataroot ./datasets/RSDDs1_cycle --model cycle_gan --name RSDDs1_cycle --load_size 256 --crop_size 256
python train.py --dataroot ./datasets/RSDDs2_cycle --model cycle_gan --name RSDDs2_cycle --load_size 256 --crop_size 256
python train.py --dataroot ./datasets/KSDD_cycle --model cycle_gan --name KSDD_cycle --load_size 512 --crop_size 512
python train.py --dataroot ./datasets/DAGM_Class1_cycle/ --model cycle_gan --name DAGM_Class1_cycle --load_size 512 --crop_size 512 --n_epochs 25 --n_epochs_decay 25 --input_nc 1 --output_nc 1 --four_rotate --netG unet_256 --iffid --batch_size 4
python train.py --dataroot ./datasets/DAGM_Class4_filted/ --model generate_defect --name DAGM_Class4_filted --load_size 512 --crop_size 512 --n_epochs 25 --n_epochs_decay 25 --input_nc 1 --output_nc 1 --four_rotate


利用cyclegan的R生成pix2pix的训练数据集，运行
python ./datasets/A_generate_B.py --datadir ./datasets/RSDDs1_cycle --modelpath ./checkpoints/RSDDs1_cycle/195_net_G_A.pth --load_size 256 --crop_size 256
python ./datasets/A_generate_B.py --datadir ./datasets/RSDDs2_cycle --modelpath ./checkpoints/RSDDs2_cycle/135_net_G_A.pth --load_size 256 --crop_size 256
python ./datasets/A_generate_B.py --datadir ./datasets/KSDD_cycle --modelpath ./checkpoints/KSDD_cycle/135_net_G_A.pth --load_size 512 --crop_size 512
python ./datasets/A_generate_B.py --datadir ./datasets/DAGM_Class1_cycle --modelpath ./checkpoints/DAGM_Class1_cycle/latest_net_G_A.pth --load_size 512 --crop_size 512

然后运行
python ./datasets/combine_A_and_B.py --fold_A ./datasets/RSDDs1_cycle_A_and_B/A --fold_B ./datasets/RSDDs1_cycle_A_and_B/B --fold_AB ./datasets/RSDDs1_pix2pix_AB
python ./datasets/combine_A_and_B.py --fold_A ./datasets/RSDDs2_cycle_A_and_B/A --fold_B ./datasets/RSDDs2_cycle_A_and_B/B --fold_AB ./datasets/RSDDs2_pix2pix_AB
python ./datasets/combine_A_and_B.py --fold_A ./datasets/KSDD_cycle_A_and_B/A --fold_B ./datasets/KSDD_cycle_A_and_B/B --fold_AB ./datasets/KSDD_pix2pix_AB

训练pix2pix
python train.py --dataroot ./datasets/RSDDs1_pix2pix_AB --model pix2pix --name RSDDs1_pix2pix --direction BtoA --load_size 256 --crop_size 256 --ifeval 1 --print_freq 5000
python train.py --dataroot ./datasets/RSDDs2_pix2pix_AB --model pix2pix --name RSDDs2_pix2pix --direction BtoA --load_size 256 --crop_size 256
python train.py --dataroot ./datasets/KSDD_pix2pix_AB --model pix2pix --name KSDD_pix2pix --direction BtoA --load_size 512 --crop_size 512

训练unet
train_unet.py

运行
python eval.py