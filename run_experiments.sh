#! /bin/bash
set -e

for trial in {1..10}; do
    cache_root=.cachestore/${RANDOM}_trial_${trial}
    CUDA_VISIBLE_DEVICES=1 python segmentation/experiments.py \
        --exp-name segmentation --trial $trial --cache-root \
        $cache_root && rm -rf $cache_root
done