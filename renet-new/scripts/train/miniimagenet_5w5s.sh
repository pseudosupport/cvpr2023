python train_5shot-new.py -batch 128 -dataset miniimagenet -gpu 6,7 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -shot 5 -milestones 40 50 -max_epoch 60
