#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env search_autoformer.py --data-set 'KSC' --indicator-name 'zico_act' --batch-size=13 --in_chans=1 --population-num 2000 --gp \
 --change_qkv  --dist-eval --cfg './experiments/search_space/space-hyper2000_sf.yaml' --output_dir './outputs/ksc'
 
