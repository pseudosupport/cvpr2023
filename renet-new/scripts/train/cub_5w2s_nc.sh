python train_original.py -batch 64 -dataset cub -gpu 0,1,2,3,4,5,6,7 -extra_dir your_run -temperature_attn 2.0 -lamb 1.5 -shot 2 -milestones 40 50 -max_epoch 60
