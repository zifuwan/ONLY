import os
import json
import random
from torch.utils.data import Dataset
from PIL import Image


class CHAIRDataset(Dataset):
    def __init__(self, data_path, anno_path, trans, model):
        self.data_path = data_path
        self.anno_path = anno_path
        self.trans = trans
        self.model = model
        
        with open(self.anno_path, 'r') as f:
            lines = f.readlines()
        coco_anns = json.loads(lines[0])

        img_dict = {}

        categories = coco_anns["categories"]
        category_names = [c["name"] for c in categories]
        category_dict = {int(c["id"]): c["name"] for c in categories}

        for img_info in coco_anns["images"]:
            img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

        for ann_info in coco_anns["annotations"]:
            img_dict[ann_info["image_id"]]["anns"].append(
                category_dict[ann_info["category_id"]
            ]
        )

        self.img_dict = img_dict
        self.img_files = os.listdir(self.data_path)
        random.shuffle(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        # img_file = 'COCO_val2014_000000185943.jpg'
        img_id = int(img_file.split(".jpg")[0][-6:])
        img_info = self.img_dict[img_id]
        assert img_info["name"] == img_file
        image_path = os.path.join(self.data_path, img_file)

        if self.model == 'llava':
            raw_image = Image.open(image_path)
            image_tensor = self.trans.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        elif self.model == 'qwen-vl':
            raw_image = Image.open(image_path).convert("RGB")
            image_tensor = self.trans(raw_image)

        elif self.model == 'instructblip':
            raw_image = Image.open(image_path).convert("RGB")
            image_tensor = self.trans['eval'](raw_image)
            
        return {"image_id": img_id, 
                "image_path": image_path, 
                "image": image_tensor}