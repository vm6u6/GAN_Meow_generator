from torch.utils.data import Dataset, DataLoader
import cv2
import os,glob,random
import torchvision.transforms as transforms
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import functools

class CatDataset(Dataset):
    def __init__(self, fnames,data_root,img_size = 64):
        

        df = pd.read_csv(fnames) 
        self.data={}
        self.data['img_path'] =  df['img_path'].values
        self.data['cat_type'] = df['cat_type'].values
        self.data['class'] = df['class'].values
        self.class_list = list(df['cat_type'].unique())
        self.data_root = data_root
        self.img_size = img_size
        self.image_shape = [img_size, img_size, 3]
        self._num_examples = len(self.data['img_path'])
        

    def num_class(self):
        return len(self.class_list)

    def read_image(self, image_file_name,size):

            img = cv2.imread(os.path.join(self.data_root,image_file_name))
            img = self.BGR2RGB(img) #because "torchvision.utils.save_image" use RGB
            transform = transforms.Compose(
                    [transforms.ToPILImage(),
                     transforms.Resize((size, size)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
            img = transform(img)
            # check its shape and reshape it to (64, 64, 3)
            return img

    def get_false_img(self, index):
        false_img_id = np.random.randint(self._num_examples)
        if false_img_id != index:
            return self.data['img_path'][false_img_id]

        return self.get_false_img(index)
    
    def one_hot_embedding(self,labels,eps=0.1):
        """Embedding labels to one-hot form.

        labes: input text
        class_list: all class list
        """
        #y = torch.eye(len(self.class_list)) 
        out  = torch.LongTensor(self.class_list.index(labels))
        #out *= 1-eps
        #out += eps/len(self.class_list)
        return out
    
    def __getitem__(self, index):

        sample = {}
        sample['true_imgs'] = torch.FloatTensor(self.read_image(self.data['img_path'][index],self.img_size))
        sample['false_imgs'] = torch.FloatTensor(self.read_image(self.get_false_img(index),self.img_size))
        sample['true_embed']  = self.data['class'][index]

        return sample 
    
    def __len__(self):
        return self._num_examples

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
