#!/bin/bash

model=resnet56
resume=checkpoints/save_1_resnet56/checkpoint.th

python -u trainer.py  --arch=$model --resume $resume -e |& tee -a log_$model
