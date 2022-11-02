gpuid=2,4
DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/miniimagenet
MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_protonet_1_pretrain/best_model.tar
cd ../../../


echo "============= meta-train 1-shot ============="
python meta_train_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 100 --milestones 40 80 --n_shot 1 --train_n_episode 1000 --val_n_episode 600 --pretrain_path $MODEL_PATH --fake 1

echo "============= meta-train 5-shot ============="
python meta_train_c.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 100 --milestones 40 80 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --pretrain_path $MODEL_PATH --fake 1