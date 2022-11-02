gpuid=0
cd ../../../

DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/miniimagenet
#DATA_ROOT=/path/mini_imagenet
#MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_5way_1shot_metatrain/best_model.tar
MODEL_2SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_8_5way_2shot_metatrain/best_model.tar

#N_SHOT=1
#python test.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_1SHOT_PATH --test_task_nums 5 --test_n_episode 2000

N_SHOT=2
python test_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_2SHOT_PATH --test_task_nums 5 --test_n_episode 2000 --fake 8

