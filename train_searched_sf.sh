#!/bin/bash

nproc_per_node=5
data_path='./dataset/cifar10'
data_set='Indian'
gp=''
change_qk=''
relative_position=''
mode='retrain'
model_type='AUTOFORMER'
dist_eval=''
cfg_base='./experiments/final/config2000/configspace-hyper2-'
output_base='./indianoutput-spacehyper100/config'
output_search='./outputs/indian/retrain_results_2000.csv'
	    
for i in {1..2000}; do
#for i in "${indx[@]}"; do
	python3 -m torch.distributed.launch \
            --nproc_per_node=$nproc_per_node \
            --use_env train.py \
            --data-path=$data_path \
            --data-set=$data_set \
            $gp $change_qk $relative_position \
            --mode $mode \
            --model_type $model_type \
            $dist_eval \
	    --world_size 5 \
            --cfg $cfg_base$i.yaml \
            --output_dir $output_base$i --epochs 300 --test_freq=20 --output_search $output_search --weight-decay 0.005 --batch-size=64 --drop=0.0 --drop-path=0.1 --in_chans=3 --change_qkv
    done
