#!/bin/bash

for model in resnet56_ftm
do
    echo "python -u trainer.py --arch=$model --lr 0.5 --save-dir=checkpoints/save_${model} |& tee results/logs/log_${model}"
    
    python -u trainer.py --arch=$model --lr 0.5 --save-dir=checkpoints/save_${model} |& tee results/logs/log_${model}
done
