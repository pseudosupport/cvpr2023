gpuid=4

DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/cub # path to the json file of CUB
cd ../../../

echo "============= pre-train ============="
python pretrain_c.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method protonet --image_size 224 --gpu ${gpuid} --lr 5e-2 --epoch 150 --milestones 100 --save_freq 100 --val meta --n_shot 1 --val_n_episode 300 --fake 8