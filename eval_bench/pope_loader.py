import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans, model):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans
        self.model = model

        if 'all' in pope_path:
            self.pope_path = [pope_path.split('all')[0] + 'random.json', pope_path.split('all')[0] + 'popular.json', pope_path.split('all')[0] + 'adversarial.json']
            image_list, query_list, label_list = [], [], []
            for p in self.pope_path:
                for q in open(p, 'r'):
                    line = json.loads(q)
                    image_list.append(line['image'])
                    query_list.append(line['text'])
                    label_list.append(line['label'])
            
        else:
            image_list, query_list, label_list = [], [], []
            for q in open(pope_path, 'r'):
                line = json.loads(q)
                image_list.append(line['image'])
                query_list.append(line['text'])
                label_list.append(line['label'])

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        
        if self.model == 'llava':
            raw_image = Image.open(image_path)
            image = self.trans.preprocess(raw_image, return_tensor='pt')['pixel_values'][0]
            query = self.query_list[index]
            label = self.label_list[index]
            return {"image": image, "query": query, "label": label, "image_path": image_path} 
            
        elif self.model == 'qwen-vl':
            raw_image = Image.open(image_path).convert("RGB")
            image = self.trans(raw_image)
            query = self.query_list[index]
            label = self.label_list[index]
            return {"image": image, "query": query, "label": label, "image_path": image_path}
        
        elif self.model == 'instructblip':
            raw_image = Image.open(image_path).convert("RGB")
            image_tensor = self.trans['eval'](raw_image)
            query = self.query_list[index]
            label = self.label_list[index]
            return {"image": image_tensor, "query": query, "label": label, "image_path": image_path}

