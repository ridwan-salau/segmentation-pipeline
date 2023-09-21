#! /bin/bash
set -e

ACQF_ARRAY=(EEIPU EI CArBO EIPS)

for trial in {1..10..2}; do

    DEVICE=0
    for acqf in ${ACQF_ARRAY[@]}; do
        # acqf="EI"
        DEVICE=$(($DEVICE%4))
        cache_root=.cachestore/${acqf}/${RANDOM}_trial_${trial}
        CUDA_VISIBLE_DEVICES=$DEVICE python segmentation/experiments.py \
            --exp-name segmentation --trial $trial --cache-root \
            $cache_root --acqf $acqf && rm -rf $cache_root &

        ((DEVICE+=1)) 
        # acqf="EEIPU"
        # cache_root=.cachestore/${acqf}/${RANDOM}_trial_$(($trial+1))
        # CUDA_VISIBLE_DEVICES=1 python segmentation/experiments.py \
        #     --exp-name segmentation --trial $(($trial+1)) --cache-root \
        #     $cache_root --acqf $acqf && rm -rf $cache_root
    done
    wait -n # Wait for the inner loop to complete before continuing
done

wait -n