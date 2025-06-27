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

from utils import dist_util
from utils.logger import create_logger
from glob import glob

import re
from PIL import Image
from torchvision.transforms import v2

from chair_loader import CHAIRDataset

# import kornia
from only_utils.only_sample import evolve_only_sampling
from only_utils.vcd_add_noise import add_diffusion_noise
evolve_only_sampling()
torch.multiprocessing.set_sharing_strategy('file_system')


import warnings
warnings.filterwarnings(action='ignore')

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
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model_path", type=str, help="model")
    parser.add_argument("--model_base", type=str, default="llava")

    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--data_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/val2014/", help="data path")
    parser.add_argument("--anno_path", type=str, default="/mnt/server17_hard1/sangmin/data/coco/annotations/instances_val2014.json")
    parser.add_argument("--log_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/logs/chair")
    parser.add_argument("--out_path", type=str, default="/mnt/server16_hard0/sangmin/code/neurips2024/chair_results/llava", help="output path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_only", type=str2bool, default=False)
    parser.add_argument("--method_name", type=str, default='none')

    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.1)

    parser.add_argument("--num_eval_samples", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--enhance_layer_index", type=int, default=0)

    args = parser.parse_known_args()[0]
    return args


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
        experiment_dir = f"{args.log_path}/{model_string_name}/{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}_{args.js_gamma}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model & Dataset
    # ========================================
    logger.info('Initializing Model')

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

    chair_dataset = CHAIRDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        trans=image_processor,
        model=args.model_base
    )
    chair_loader = DataLoader(
        chair_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    os.makedirs(
        args.out_path, exist_ok=True
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

    for batch_id, data in tqdm(enumerate(chair_loader), total=args.num_eval_samples):

        # early stop for debuggging purpose
        # if batch_id == 20:
        #     break

        if batch_id == args.num_eval_samples:
            break
            
        img_id = data["image_id"]
        image_path = data["image_path"]
        image = data["image"][0]


        qs =  "Please describe this image in detail."

        
        image_pos = None
        image_neg = None
        
        if args.use_ritual:
            # ==============================================
            #              Image Transforms
            # ==============================================
            raw_image = Image.open(image_path[0])
            pos_aug = random.choice(list(aug_dict.keys()))

            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](raw_image)
                image_pos = vis_processors['eval'](raw_image_pos) 
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
                with torch.no_grad():
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
                    #     max_new_tokens=args.max_new_tokens,
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
                    #     js_gamma=args.js_gamma,
                    # )
                    output_ids, _ = model.generate(
                        input_ids=input_ids.input_ids.cuda(),
                        attention_mask=input_ids.attention_mask.cuda(),
                        images=image.unsqueeze(0).half().cuda(),
                        images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                        images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=1,
                        min_new_tokens=args.max_new_tokens,
                        # length_penalty=1,
                        num_return_sequences=1,
                        output_hidden_states=False,
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

        outputs = [
            tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in output_ids
        ][0]
        # outputs = response
        outputs = outputs.strip()
        
        logger.info(f"[VQA for ritual]")
        logger.info(f"V: {image_path}")
        logger.info(f"Q: {qs}")
        logger.info(f"A: {outputs}")
        logger.info(f"="*50)

        img_save = {}
        img_save["image_id"] = img_id.item()
        img_save["caption"] = outputs

        # dump metric file
        use_method = 'use_' + args.method_name
        methods = {'use_ritual': args.use_ritual, 'use_vcd': args.use_vcd, 'use_m3id': args.use_m3id, 'use_only': args.use_only, 'use_none': True}
        if methods[use_method]!=True:
            print("Please check the method name!!!!!!")
        with open(os.path.join(args.out_path, f"{args.ritual_alpha_pos}_{args.ritual_alpha_neg}_{args.ritual_beta}_{args.js_gamma}_{args.max_new_tokens}_{args.method_name}.jsonl"), "a") as f:
            json.dump(img_save, f)
            f.write('\n')
    
    logger.info(vars(args))

    if args.use_ritual:
        logger.info(f"RITUAL Transformation: {pos_aug_counter}")

if __name__ == "__main__":
    main()