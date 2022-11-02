gpuid=3
DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/miniimagenet
MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_8_pretrain/best_model.tar
cd ../../../


#echo "============= meta-train 1-shot ============="
#python meta_train_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 100 --milestones 40 80 --n_shot 1 --train_n_episode 1000 --val_n_episode 600 --pretrain_path $MODEL_PATH --fake 8
echo "============= meta-train 2-shot ============="
python meta_train_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 100 --milestones 40 80 --n_shot 2 --train_n_episode 900 --val_n_episode 600 --pretrain_path $MODEL_PATH --fake 8

echo "============= meta-train 3-shot ============="
python meta_train_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 100 --milestones 40 80 --n_shot 3 --train_n_episode 800 --val_n_episode 600 --pretrain_path $MODEL_PATH --fake 8

echo "============= meta-train 4-shot ============="
python meta_train_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 100 --milestones 40 80 --n_shot 4 --train_n_episode 700 --val_n_episode 600 --pretrain_path $MODEL_PATH --fake 8

#echo "============= meta-train 5-shot ============="
#python meta_train_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 100 --milestones 40 80 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH --fake 8