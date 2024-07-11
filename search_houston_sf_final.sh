#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env search_autoformer.py --data-set 'Houston' --indicator-name 'zico_act' --batch-size=15 --in_chans=3 --population-num 2000 --gp \
 --change_qkv  --dist-eval --cfg './experiments/search_space/space-hyper2000_sf.yaml' --output_dir './outputs/houston'
