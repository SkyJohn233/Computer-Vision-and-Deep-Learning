#encoding=utf-8
import resnet34
from data import TrainDataSet,TestDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from torch import optim
import torch.nn as nn
import torch.cuda
import torch
import os


batch_size = 4

def val(model, dataloader):
	model.eval()
	eval_loss = 0
	eval_acc = 0
	for i,data in enumerate(dataloader,0):
		inputs,labels = data
		inputs,labels = V(inputs,volatile=True), V(labels,volatile=True)
		if use_cuda:
			inputs = inputs.cuda()
			labels = labels.cuda()
		outputs = net(inputs)
		if use_cuda:
			criterion.cuda()
		loss_eval = criterion(outputs, labels)
		eval_loss += loss_eval.data[0]
		for i in range(batch_size):
			t_label = labels[i]
			sorted_,indices = torch.topk(outputs[i],3,0)
			for _ in range(3):
				if int(t_label) == int(indices[_]):
					eval_acc = eval_acc + 1
	return float(eval_loss/len(dataloader)), float(eval_acc/len(dataloader)/batch_size)



use_cuda = torch.cuda.is_available()
net = resnet34.ResNet(80)
if use_cuda:
	net.cuda()
net.train()
root_ = os.path.abspath(os.curdir)
traindataset = TrainDataSet(root_)
testdataset = TestDataSet(root_)
dataloader = DataLoader(traindataset, batch_size =4,shuffle=True, num_workers=40,pin_memory=True,drop_last=True)
test_dataloader = DataLoader(testdataset,batch_size=4,shuffle=True,num_workers=40,pin_memory=True,drop_last=True)
epoch_num = 25
criterion = nn.CrossEntropyLoss()
optimizer =  optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

for epoch in range(epoch_num):
	net.train()
	running_loss = 0.0
	for i,data in enumerate(dataloader,0):
		inputs,labels = data
		if use_cuda:
			inputs, labels = V(inputs.cuda()), V(labels.cuda())
		else:
			inputs, labels = V(inputs), V(labels)
		inputs.cuda()
		labels.cuda()
		optimizer.zero_grad()
		outputs = net(inputs)
		if use_cuda:
			criterion.cuda()
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.data[0]
		if i % 2000 == 1999:
			print("[%d %d] loss:%.3f " % (epoch+1, i+1, running_loss/2000))
			running_loss = 0.0
	train_loss, train_acc = val(net,dataloader)
	print("train loss: %.5f train_acc %.5f" %(train_loss, train_acc))
	test_loss, test_acc = val(net,test_dataloader)
	print("test loss: %.5f test_acc %.5f" %(test_loss,test_acc))
	torch.save(net,"resnet34_"+str(epoch)+".pkl")

print("finish training")
torch.save(net,"resnet34.pkl")
