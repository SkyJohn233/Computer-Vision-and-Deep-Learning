#encoding=utf-8
import resnet34
from submit import ValDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from torchvision.utils import make_grid, save_image
import shutil
from torch import optim
import torch.nn as nn
import torch.cuda
import torch
import os

batch_size = 4
trans_2 = T.Compose([T.ToTensor()])

use_cuda = torch.cuda.is_available()
modelpath=r"resnet34_20.pkl"
net = torch.load(modelpath)
if use_cuda:
	net.cuda()
else:
	net.cpu()
root_ = os.path.abspath(os.curdir)
valdataset = ValDataSet(root_)
net.eval()
f = open(root_+"/submit.info","w",encoding="utf-8")
for i in valdataset.imgs:
	inputs = trans_2(Image.open(valdataset.root+"/transform_val/"+i))
	inputs = V(torch.unsqueeze(inputs,0))
	if use_cuda:
		inputs = inputs.cuda()
	outputs = net(inputs)
	sorted_, indices = torch.topk(outputs[0],3,0)
	f.write("test/"+i+" ")
	for k in range(3):
		f.write(str(int(indices[k]))+" ")
	print(i, indices)
	f.write("\n")

f.close()
