#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import os
import json

from PIL import Image

import argparse


# In[ ]:


parser = argparse.ArgumentParser(description="Predict")
parser.add_argument('--image_path', type=str, required=True, help="C:\Users\Nayara\Desktop\AI and ML\flowers102\train")
parser.add_argument('--checkpoint_path', type=str, required=True, help='C:/Users/Nayara/Desktop/AI and ML/flowers102/checkpoint.pth')
parser.add_argument('--top_k', type=int, default=5, help='top K classes')
parser.add_argument('--category_names', type=str, help="C:\Users\Nayara\Desktop\AI and ML\flowers102\cat_to_name.json")
parser.add_argument('--gpu', action='store_true', help='GPU')

args = parser.parse_args()


# In[6]:


main_dir = 'C:\\Users\\Nayara\\Desktop\\AI and ML\\flowers102'

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(os.path.join(main_dir, 'train'), transform=transform)
valid_data = datasets.ImageFolder(os.path.join(main_dir, 'val'), transform=transform)
test_data = datasets.ImageFolder(os.path.join(main_dir, 'test'), transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)


# In[ ]:


model = load_checkpoint('C:/Users/Nayara/Desktop/AI and ML/flowers102/checkpoint.pth')
probabilities, classes = predict('C:/Users/Nayara/Desktop/AI and ML/flowers102/test/7/image_07223.jpg', model)
display_prediction(image_path, model, cat_to_name, topk=5)


# In[ ]:


if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[str(class_)] for class_ in classes[0].tolist()]

# Imprimindo os resultados
print("Predict:", classes)
print("Probs:", probs[0].tolist())

