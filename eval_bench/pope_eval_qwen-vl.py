import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
# print(sys.path)

from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM
from Qwen_VL.modeling_qwen import QWenLMHeadModel

from utils import dist_util
from utils.logger import create_logger
from glob import glob

import re
from PIL import Image
from torchvision.transforms import v2

from pope_loader import POPEDataSet

from only_utils.only_sample import evolve_only_sampling
from only_utils.vcd_add_noise import add_diffusion_noise
evolve_only_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, default="/mnt/server8_hard1/donguk/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--data_path", type=str, default="/mnt/server18_hard0/jhjang/LVLM/crg/data/coco/val2014")
    parser.add_argument("--pope_path", type=str, default="/mnt/server8_hard1/donguk/rips2024/experiments/data/POPE/coco/coco_pope_random.json")
    parser.add_argument("--log_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/logs/pope")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_only", type=str2bool, default=False)
    parser.add_argument("--enhance_layer_index", type=int, default=0)

    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.1)

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--experiment_index", type=int, default=0)
    parser.add_argument("--type", type=str, default="random")
    parser.add_argument("--dataset_name", type=str, default="coco")

    args = parser.parse_args()
    return args


def print_acc(pred_list, label_list, logger):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, precision, recall, f1, yes_ratio


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out.split('\n'):

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
            break
        else:
            pred_list.append(1)
            break
    
    return pred_list


def main():
    args = parse_args()
    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        # model_string_name = args.model_path.split("/")[-1]
        model_string_name = 'qwen-vl'
        if args.use_ritual:
            method_name = "RITUAL"
        elif args.use_vcd:
            method_name = "VCD"
        elif args.use_m3id:
            method_name = "M3ID"
        elif args.use_only:
            method_name = "ONLY"
        else:
            method_name = "Regular"
        # experiment_index = len(glob(f"{args.log_path}/{model_string_name}/*")) + args.experiment_index
        # experiment_index = args.experiment_index
        experiment_dir = f"{args.log_path}/pope/{model_string_name}/{method_name}_{args.dataset_name}_{args.type}_{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}_{args.js_gamma}_layer_{args.enhance_layer_index}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    print('Initializing Model')

    #### for ritual
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()
    image_processor = model.transformer.visual.image_transform

    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path,
        trans=image_processor,
        model=args.model_base
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    # ==============================================
    #               Augmentations
    # ==============================================

    aug_dict = {
    'horizontal flip':v2.RandomHorizontalFlip(p=1),
    'vertical flip':v2.RandomVerticalFlip(p=1),
    'rotation':v2.RandomRotation(degrees=180),
    'color jitter':v2.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
    'gaussian blur':v2.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
    'crop':v2.RandomResizedCrop(size=336),
    }
    
    # For statistics
    pos_aug_counter = {k:0 for k in aug_dict}
    pos_aug_counter.update({None: 0})

    # ========================================
    #            Start Generation
    # ========================================
    print("Start eval...")
    pred_list, label_list = [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"][0]
        qs = data["query"][0]
        label = data["label"]
        image_path = data["image_path"]
        label_list = label_list + list(label)
        
        image_pos = None
        image_neg = None
        
        if args.use_ritual:
            # ==============================================
            #              Image Transforms
            # ==============================================
            raw_image = Image.open(image_path).convert("RGB")
            pos_aug = random.choice(list(aug_dict.keys()))

            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                image_pos = image_processor['eval'](raw_image_pos)
                image_pos = torch.tensor(image_pos).unsqueeze(0)
            
            pos_aug_counter[pos_aug] += 1
            print(f"RITUAL Transformation: {pos_aug}")
        
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image, args.noise_step)
            
        # ==============================================
        #              Text prompt setting
        # ==============================================
        qu_out = '<img>{}</img>{} Answer:'.format(image_path[0], qs)
        
        input_ids = tokenizer([qu_out], return_tensors='pt', padding='longest')
        query_out = tokenizer.from_list_format([
            {'image': image_path[0]},
            {'text': qs},
        ])

        with torch.inference_mode():
            with torch.inference_mode():
                # response, _ = model.chat(
                #     tokenizer=tokenizer,
                #     query=query_out,
                #     history=None,
                #     # attention_mask=input_ids.attention_mask.cuda(),
                #     # images=image.unsqueeze(0).cuda(),
                #     images_pos=(image_pos.unsqueeze(0).cuda() if image_pos is not None else None),
                #     images_neg=(image_neg.unsqueeze(0).cuda() if image_neg is not None else None),
                #     do_sample=True,
                #     temperature=args.temperature,# args.temperature
                #     top_p=args.top_p,
                #     top_k=args.top_k,
                #     # min_new_tokens=1,
                #     # length_penalty=1,
                #     # num_return_sequences=1,
                #     max_new_tokens=6,
                #     output_hidden_states=False,
                #     use_cache=True,
                #     pad_token_id=tokenizer.eod_id,
                #     eos_token_id=tokenizer.eod_id,
                #     use_ritual=args.use_ritual,
                #     use_vcd=args.use_vcd,
                #     use_m3id=args.use_m3id,
                #     use_only=args.use_only,
                #     enhance_layer_index=args.enhance_layer_index,
                #     ritual_alpha_pos=args.ritual_alpha_pos,
                #     ritual_alpha_neg=args.ritual_alpha_neg,
                #     ritual_beta=args.ritual_beta,
                # )
                response, overlapping_index_len = model.generate(
                    input_ids=input_ids.input_ids.cuda(),
                    attention_mask=input_ids.attention_mask.cuda(),
                    images=image.unsqueeze(0).cuda(),
                    images_pos=(image_pos.unsqueeze(0).cuda() if image_pos is not None else None),
                    images_neg=(image_neg.unsqueeze(0).cuda() if image_neg is not None else None),
                    do_sample=True,
                    temperature=args.temperature,# args.temperature
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=8,
                    min_new_tokens=1,
                    length_penalty=1,
                    num_return_sequences=1,
                    output_hidden_states=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eod_id,
                    eos_token_id=tokenizer.eod_id,
                    use_ritual=args.use_ritual,
                    use_vcd=args.use_vcd,
                    use_m3id=args.use_m3id,
                    use_only=args.use_only,
                    enhance_layer_index=args.enhance_layer_index,
                    ritual_alpha_pos=args.ritual_alpha_pos,
                    ritual_alpha_neg=args.ritual_alpha_neg,
                    ritual_beta=args.ritual_beta,
                    js_gamma=args.js_gamma,
                )
                    
        response = [
            tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in response
        ][0]
        outputs = response.strip()
        pred_list = recorder(outputs, pred_list)

        print(f"[VQA for ritual]")
        print(f"V: {image_path}")
        print(f"Q: {qs}")
        print(f"A: {outputs}")
        if label == 1: print(f"GT: Yes")
        elif label == 0: print(f"GT: No")
        print(f"="*50)
        if pred_list[-1] == label_list[-1]:
            print("Correct")
        else:
            print('Wrong')

        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        # import ipdb; ipdb.set_trace()
        
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        
        print(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )

    if len(pred_list) != 0:
        logger.info(vars(args))
        # logger.info("Prompt for Aug:", prompt_aug)
        # logger.info("Prompt for ritual:", prompt_out)
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list, logger)
        
        acc = round(acc*100,2)
        precision = round(precision*100,2)
        recall = round(recall*100,2)
        f1 = round(f1*100,2)
        yes_ratio = round(yes_ratio*100,2)
        
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )
        # if args.use_ritual:
        #     logger.info(f"RITUAL Transformation: {pos_aug_counter}")

if __name__ == "__main__":
    main()
