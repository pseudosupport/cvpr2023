gpuid=4
DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/miniimagenet

#MODEL_1SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_6_5way_1shot_metatrain/best_model.tar
MODEL_4SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_6_5way_4shot_metatrain/best_model.tar
#MODEL_5SHOT_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_6_5way_5shot_metatrain/best_model.tar
cd ../../../

#N_SHOT=1
#python test_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_1SHOT_PATH --test_task_nums 5 --test_n_episode 2000 --fake 6

N_SHOT=4
python test_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_4SHOT_PATH --test_task_nums 5 --test_n_episode 2000 --fake 6

#N_SHOT=5
#python test_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --n_shot $N_SHOT --reduce_dim 640 --model_path $MODEL_5SHOT_PATH --test_task_nums 5 --test_n_episode 2000 --fake 6
