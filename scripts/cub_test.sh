#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..
nvidia-smi

MODEL=dpn
DATANAME=cub
BACKBONE=resnet101
SAVEPATH=./output/${DATANAME}/bs-256+v2s1.0+attri_norm+att_weight1e3
MODELPATH=./output/cub/bs-256+v2s1.0+attri_norm+att_weight1e3/dpn_0.7218.model
DATAPATH='../../ZSL-Dataset/CUB_200_2011/images'


STAGE1=1

if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 64 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --is_fix \
    --resume ${MODELPATH} \
    --eval_only
fi