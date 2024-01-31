#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--data_dir', type=str, default='C:\\Users\\Nayara\\Desktop\\AI and ML\\flowers102', help='train data')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='60 epochs')

args = parser.parse_args()
data_dir = args.data_dir
batch_size = args.batch_size
num_epochs = args.num_epochs

import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


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


# In[3]:


model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 102)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[4]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*28*28, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 102)
        self.bn5 = nn.BatchNorm1d(102)

    def forward(self,x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = torch.dropout(x, 0.5, train=self.training)
        x = self.fc2(x)
        return x


# In[5]:


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


num_epochs = 60
losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.2f}")

    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (pred == labels).sum().item()

    train_accuracy = (correct_train / total_train) * 100
    train_accuracies.append(train_accuracy)

    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (pred == labels).sum().item()

    val_accuracy = (correct_val / total_val) * 100
    val_accuracies.append(val_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}%")
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%")


# In[ ]:


def save_checkpoint(model, filepath, class_to_idx):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, filepath)


# In[ ]:


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = CNN()
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model    


# In[ ]:


def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image)
    return image


# In[ ]:


def predict(image_path, model, topk=5):
    processed_image = process_image(image_path) 
    processed_image = processed_image.unsqueeze(0)
    model.eval()

    with torch.no_grad():
        outputs = model(processed_image)  
        probs = torch.exp(outputs)
        top_probs, top_classes = probs.topk(topk, dim=1)

    return top_probs, top_classes


# In[ ]:


def display_prediction(image_path, model, cat_to_name, topk=5):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Erro ao abrir a imagem: {e}")
        return
    
    probs, classes = predict(image_path, model, topk=topk)
    probs = probs[0].tolist()
    classes = classes[0].tolist()

    class_names = [cat_to_name[str(cls)] for cls in classes]

    image = Image.open(image_path)
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)

    ax.imshow(image)
    ax.axis('off')

    plt.subplot(2, 1, 2)
    plt.barh(class_names, probs)
    plt.xlabel('Probabilidade')
    plt.title('Top K Classes Previstas')
    plt.show()
    


# In[ ]:


save_checkpoint(model, 'C:/Users/Nayara/Desktop/AI and ML/flowers102/checkpoint.pth', train_data.class_to_idx)


# In[ ]:


json_string = """
{
    "21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose",
    "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", 
    "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip",
    "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox",
    "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood",
    "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower",
    "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation",
    "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower",
    "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea",
    "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower",
    "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy",
    "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm",
    "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold",
    "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya",
    "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily",
    "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower",
    "52": "wild pansy", "84": "columbine", "12": "colts foot", "11": "snapdragon", "96": "camellia",
    "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea",
    "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower",
    "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium",
    "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus",
    "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose",
    "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"
}
"""
cat_to_name = json.loads(json_string)


# In[ ]:


checkpoint = torch.load('C:/Users/Nayara/Desktop/AI and ML/flowers102/checkpoint.pth')

