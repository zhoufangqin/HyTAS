#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env search_autoformer.py --data-set 'Indian' --indicator-name 'snip' --batch-size=16 --in_chans=3 --population-num 2000 --gp \
 --change_qkv  --dist-eval --cfg './experiments/search_space/final/space-hyper2000_sf.yaml' --output_dir './outputs/indian'
 
