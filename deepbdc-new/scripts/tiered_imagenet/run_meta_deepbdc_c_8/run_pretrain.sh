gpuid=4

DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/tieredimagenet # path to the json file of CUB
cd ../../../

echo "============= pre-train ============="
python pretrain_c.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --wd 1e-4 --epoch 100 --milestones 40 70 --save_freq 50 --reduce_dim 256 --dropout_rate 0.6 --val meta --val_n_episode 1000 --fake 8
