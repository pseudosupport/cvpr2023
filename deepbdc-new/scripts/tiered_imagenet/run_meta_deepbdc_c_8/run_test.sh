gpuid=2
N_SHOT=2

DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/tieredimagenet # path to the json file of CUB
MODEL_PATH=./checkpoints/tiered_imagenet/ResNet12_meta_deepbdc_8_5way_${N_SHOT}shot_metatrain/best_model.tar
cd ../../../

python test_c.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 --fake 8
