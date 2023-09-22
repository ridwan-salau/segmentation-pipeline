#! /bin/bash
set -e

ACQF_ARRAY=("EEIPU")

for trial in {1..2..2}; do
    for acqf in ${ACQF_ARRAY[@]}; do
        # acqf="EI"
        # cache_root=.cachestore/${acqf}/${RANDOM}_trial_${trial}
        # CUDA_VISIBLE_DEVICES=0 python segmentation/experiments.py \
        #     --exp-name segmentation --trial $trial --cache-root \
        #     $cache_root --acqf $acqf && rm -rf $cache_root &

        acqf="CArBO"
        cache_root=.cachestore/${acqf}/${RANDOM}_trial_$(($trial+1))
        CUDA_VISIBLE_DEVICES=1 python segmentation/experiments.py \
            --exp-name segmentation --trial $(($trial+1)) --cache-root \
            $cache_root --acqf $acqf && rm -rf $cache_root
    done
done

wait