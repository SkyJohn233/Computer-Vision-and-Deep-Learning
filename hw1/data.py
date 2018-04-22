#encoding = utf-8

## training-dataset ##

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

class TestDataSet(data.Dataset):
	def __init__(self, root, transform=None):
		self.root = root
		self.transforms = transform
		self.num = 0
		new_root = self.root + "/transform_test/"
		self.imgs = os.listdir(new_root)


	def __getitem__(self, index):
		tt = self.imgs[index]
		label = int(tt.split("_")[0])
		new_root = self.root + "/transform_test/"
		data = Image.open(os.path.join(new_root,tt))
		data = trans_2(data)
		return data,label

	def __len__(self):
		return len(self.imgs)


class TrainDataSet(data.Dataset):
	def __init__(self, root, transform=None):
		self.root = root
		self.transforms = transform
		self.num = 0
		new_root = self.root + "/transform_train/"
		self.imgs = os.listdir(new_root)


	def __getitem__(self, index):
		tt = self.imgs[index]
		label = int(tt.split("_")[0])
		new_root = self.root + "/transform_train/"
		data = Image.open(os.path.join(new_root,tt))
		data = trans_2(data)
		return data,label

	def __len__(self):
		return len(self.imgs)

	def transform_single_data(self, path_, label):
		img_path = os.path.join(self.root, path_)
		image = Image.open(img_path)
		if self.transforms:
			image = self.transforms(image)
		new_img_path = self.root + "/transform_train/" + label +"_" + path_.split("/")[1]
		save_image(image, new_img_path)
		self.num = self.num + 1
		print("finish " , self.num)


	def transform_all_data(self):
		with open("train.info","r",encoding="utf-8") as f:
			for line in f.readlines():
				tt = line.split(' ')
				path_, label = tt[0], tt[1]
				self.transform_single_data(path_, label)
		print("finish all")

if __name__ == "__main__":
	#os.mkdir(root_+"/transform_train")
	traindataset = TrainDataSet(root_, trans_)
	#traindataset.transform_all_data()
	#os.mkdir(root_+"/transform_test")
	testdataset =TestDataSet(root_)
	total_img_num = len(traindataset.imgs)
	print(total_img_num)
	for _ in range(int(0.3*total_img_num)):
		i = random.randint(0,len(traindataset.imgs)-1)
		print("delete ",i)
		testdataset.imgs.append(traindataset.imgs[i])
		tt = traindataset.imgs[i]
		new_root = traindataset.root + "/transform_train"
		image_path = os.path.join(new_root,tt)
		shutil.move(image_path,testdataset.root+"/transform_test")
		traindataset.imgs.pop(i)



