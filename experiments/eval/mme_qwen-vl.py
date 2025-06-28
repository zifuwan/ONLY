import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM

from Qwen_VL.modeling_qwen import QWenLMHeadModel
# import kornia
from transformers import set_seed
from only_utils.only_sample import evolve_only_sampling
from only_utils.vcd_add_noise import add_diffusion_noise
evolve_only_sampling()

from torchvision.transforms import v2
import random 

import warnings
warnings.filterwarnings(action='ignore')

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def eval_model(args):
    # Model
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

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    for line in tqdm(questions):
    # for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        # cur_prompt = qs

        # input_ids = tokenizer_image_token(qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        # import pdb; pdb.set_trace()
        
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(os.path.join(args.image_folder, image_file))
        image = image_processor(image)
        
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

        image_pos = None
        image_neg = None
    
        
        if args.use_ritual:
            pos_aug = random.choice(list(aug_dict.keys()))

            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](image)
                image_pos = vis_processors['eval'](raw_image_pos).unsqueeze(0)
                image_pos = torch.tensor(image_pos)
            
            pos_aug_counter[pos_aug] += 1
            print(f"RITUAL Transformation: {pos_aug}")
        
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image_tensor, args.noise_step)
            
        qu_out = '<img>{}</img>{} Answer:'.format(image_path, qs)
        
        input_ids = tokenizer([qu_out], return_tensors='pt', padding='longest')
        
        with torch.inference_mode():
            response, _ = model.generate(
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
        
        print(f"Question: {qs}")
        print(f"Answer: {outputs}")
        
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    parser.add_argument("--use_only", type=str2bool, default=False)
    
    parser.add_argument("--ritual_alpha_pos", type=float, default=3)
    parser.add_argument("--ritual_alpha_neg", type=float, default=1)
    parser.add_argument("--ritual_beta", type=float, default=0.1)
    parser.add_argument("--js_gamma", type=float, default=0.1)
    parser.add_argument("--enhance_layer_index", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
