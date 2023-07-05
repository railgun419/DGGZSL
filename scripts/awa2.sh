#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..
nvidia-smi

MODEL=dpn
ATTEMBDIM=25
DATANAME=AWA2
BACKBONE=resnet101
DATAPATH=
SAVEPATH=./output/${DATANAME}
RESNETPRE=

STAGE1=1

if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 256 \
    --lr 2e-4 \
    --epochs 60 \
    --w_L_v2s 80 \
    --T 0.1 \
    --K 10 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --save_H_OPT \
    --is_fix \
    --att-emb-dim ${ATTEMBDIM} \
    --seed 9007 \
    --seed1 9007 \
    --seed2 5224 \
    --resnet_pretrain ${RESNETPRE}
fi
#
#if [ ${STAGE2} = 1 ]
#then
#  python main.py \
#    --batch-size 64 \
#    --lr 2e-4 \
#    --epochs 60 \
#    --backbone ${BACKBONE} \
#    --model ${MODEL} \
#    --data-name ${DATANAME} \
#    --save-path ${SAVEPATH} \
#    --data ${DATAPATH} \
#    --resume ${SAVEPATH}/fix.model \
#    --att-emb-dim ${ATTEMBDIM}  \
#    --seed 414 \
#    --seed1 8375 \
#    --seed2 5004 \
#    --resnet_pretrain ${RESNETPRE}
#fi
