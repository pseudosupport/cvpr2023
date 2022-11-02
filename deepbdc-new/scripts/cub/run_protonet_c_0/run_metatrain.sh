gpuid=4

DATA_ROOT=/home/chenxu/mit2023/cx_work/DEEPBDC-new/filelist/cub # path to the json file of CUB
MODEL_PATH=./checkpoints/cub/ResNet18_protonet_0_pretrain/best_model.tar
cd ../../../


echo "============= meta-train 2-shot ============="
python meta_train_c.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method protonet --image_size 224 --gpu ${gpuid} --lr 1e-3 --epoch 100 --milestones 40 80 --n_shot 2 --train_n_episode 600 --val_n_episode 300 --pretrain_path $MODEL_PATH --fake 0

#echo "============= meta-train 5-shot ============="
#python meta_train.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method protonet --image_size 224 --gpu ${gpuid} --lr 1e-3 --epoch 100 --milestones 40 80 --n_shot 5 --train_n_episode 600 --val_n_episode 300 --pretrain_path $MODEL_PATH
