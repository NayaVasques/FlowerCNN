#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

main_dir = 'C:\\Users\\Nayara\\Desktop\\AI and ML\\flowers102'
subsets = ['train', 'val', 'test']

for subset in subsets:
    for i in range(1, 103):  # 102 classes
        path = os.path.join(main_dir, subset, str(i))
        os.makedirs(path, exist_ok=True)

print("Diretórios criados com sucesso!")


# In[2]:


import scipy.io
import shutil
import os

labels_mat = 'C:\\Users\\Nayara\\Desktop\\AI and ML\\flowers102\\imagelabels.mat'
setid_mat = 'C:\\Users\\Nayara\\Desktop\\AI and ML\\flowers102\\setid.mat'
images_dir = 'C:\\Users\\Nayara\\Desktop\\AI and ML\\flowers102\\jpg'  

labels = scipy.io.loadmat(labels_mat)['labels'][0]
split = scipy.io.loadmat(setid_mat)
train_idx, val_idx, test_idx = split['trnid'][0], split['valid'][0], split['tstid'][0]

def organize_images(indices, subset):
    for idx in indices:
        image_name = f'image_{idx:05d}.jpg'
        label = labels[idx - 1]
        source = os.path.join(images_dir, image_name)
        destination = os.path.join(main_dir, subset, str(label), image_name)
        shutil.move(source, destination)

main_dir = 'C:\\Users\\Nayara\\Desktop\\AI and ML\\flowers102'
organize_images(train_idx, 'train')
organize_images(val_idx, 'val')
organize_images(test_idx, 'test')

print("Imagens organizadas com sucesso!")

