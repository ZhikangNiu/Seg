CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 train.py --batch_size 8 --niter 300 --lr 3e-5 --resume > train.log