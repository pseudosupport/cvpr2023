gpuid=1
N_SHOT=2

DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/cub # path to the json file of CUB
MODEL_PATH=./checkpoints/cub/ResNet18_meta_deepbdc_8_5way_${N_SHOT}shot_metatrain/best_model.tar
cd ../../../

python test_c.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 256 --model_path $MODEL_PATH --test_task_nums 5 --fake 8
