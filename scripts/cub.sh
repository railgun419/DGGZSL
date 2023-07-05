#!/bin/bash
script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}
cd ..
nvidia-smi

MODEL=dpn
DATANAME=CUB
BACKBONE=resnet101
DATAPATH=
SAVEPATH=./output/${DATANAME}
RESNETPRE=


STAGE1=1
STAGE2=1
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2--master_port 12345 main.py
if [ ${STAGE1} = 1 ]
then
  python main.py \
    --batch-size 128 \
    --lr 2e-4 \
    --epochs 60 \
    --T 0.5 \
    --K 30 \
    --w_L_v2s 300 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --is_fix \
    --seed 8432 \
    --seed1 5554 \
    --seed2 9169 \
    --resnet_pretrain ${RESNETPRE}
fi

if [ ${STAGE2} = 1 ]
then
  python main.py \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 60 \
    --T 0.5 \
    --K 30 \
    --w_L_v2s 300 \
    --backbone ${BACKBONE} \
    --model ${MODEL} \
    --data-name ${DATANAME} \
    --save-path ${SAVEPATH} \
    --data ${DATAPATH} \
    --resume ${SAVEPATH}/fix.model \
    --seed 414 \
    --seed1 8375 \
    --seed2 5004 \
    --resnet_pretrain ${RESNETPRE}
fi
