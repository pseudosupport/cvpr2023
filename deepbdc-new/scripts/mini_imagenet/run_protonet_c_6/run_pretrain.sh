gpuid=4

DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/miniimagenet
cd ../../../

echo "============= pre-train ============="

python pretrain_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-2 --wd 1e-4 --epoch 170 --milestones 100 150 --save_freq 100 --val meta --val_n_episode 600 --n_shot 5 --fake 6