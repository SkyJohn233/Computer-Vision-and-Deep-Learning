#encoding=utf-8

## val-datset
import os
import numpy as np 
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from torchvision.utils import make_grid, save_image
import shutil
import random

trans_ = T.Compose([T.Resize(224),
					T.CenterCrop(224),
					T.ToTensor(),
					T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])])

trans_2 = T.Compose([T.ToTensor()])
root_ = r"/home/skyjohn/university/CVDL/hw1"

class ValDataSet(data.Dataset):
	def __init__(self, root, transform=None):
		self.root = root
		self.transforms = transform
		self.num = 0
		new_root = self.root + "/transform_val/"
		self.imgs = os.listdir(new_root)


	def __getitem__(self, index):
		tt = self.imgs[index]
		label = int(tt.split("_")[0])
		new_root = self.root + "/transform_val/"
		data = Image.open(os.path.join(new_root,tt))
		data = trans_2(data)
		return data,label

	def __len__(self):
		return len(self.imgs)

	def transform_single_data(self, path_):
		img_path = os.path.join(self.root, path_)
		image = Image.open(img_path)
		if self.transforms:
			image = self.transforms(image)
		new_img_path = self.root + "/transform_val/"  + path_.split("/")[1]
		save_image(image, new_img_path)
		self.num = self.num + 1
		print("finish " , self.num)


	def transform_all_data(self):
		new_root_ = root_ + "/test"
		img_paths = os.listdir(new_root_)
		for i in img_paths:
			self.transform_single_data("test/"+i)
		print("finish all")

if __name__=="__main__":
	os.mkdir(root_+"/transform_val")
	valdataset=ValDataSet(root_,trans_)
	valdataset.transform_all_data()