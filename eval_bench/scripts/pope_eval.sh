#!/bin/bash

seed=42
dataset_name="coco" # coco | aokvqa | gqa
type="adversarial" # random | popular | adversarial

# # llava
model="llava"
model_path="/data/ce/model/llava-v1.5-7b"

# instructblip
# model="instructblip"
# model_path=None

# qwen-vl
# model="qwen-vl"
# model_path="/data/zifu/model/Qwen-VL-Chat"

pope_path="/data/ce/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json"
data_path="/data/ce/data/coco/val2014"

# data_path="/data/ce/data/gqa/images"

log_path="./logs"

use_ritual=False
use_vcd=False
use_m3id=False
use_only=True

ritual_alpha_pos=3.0
ritual_alpha_neg=1.0
ritual_beta=0.1
js_gamma=0.2
enhance_layer_index=0


#####################################
# Run single experiment
#####################################
export CUDA_VISIBLE_DEVICES=4
python eval_bench/pope_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--pope_path ${pope_path} \
--data_path ${data_path} \
--log_path ${log_path} \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--use_only ${use_only} \
--ritual_alpha_pos ${ritual_alpha_pos} \
--ritual_alpha_neg ${ritual_alpha_neg} \
--ritual_beta ${ritual_beta} \
--js_gamma ${js_gamma} \
--type ${type} \
--dataset_name ${dataset_name} \
--enhance_layer_index ${enhance_layer_index} \
