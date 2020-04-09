
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from model_registrar import ModelRegistrar
from cifar import CIFAR

# 数据处理
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
										download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=12,
										  shuffle=True, num_workers=0)

		   
		   
# 模型构建
device = torch.device('cpu')
model_dir = './checkpoints'
model_registrar = ModelRegistrar(model_dir, device) # 注册模型 完成对模型的一些基本操作

# model_registrar.load_models(0)

model_1 = CIFAR(model_registrar,device)
model_1.create_graphical_model()
model_2 = CIFAR(model_registrar,device)
model_2.create_graphical_model()

# print(model_registrar.print_model_names())

#optimizer = optim.SGD(model_registrar.parameters(), lr=0.001, momentum=0.9) 
lr = 0.01
wd = 0.01
optimizer_w = optim.SGD([{"params":param} for name,param in model_registrar.named_parameters() if 'weight' in name], lr=lr, weight_decay=wd) # 对权重参数衰减
optimizer_b = optim.SGD([{"params":param} for name,param in model_registrar.named_parameters() if 'bias' in name], lr=lr)  # 不对偏差参数衰减

# 开始训练

count = 0

plt.ion()
fig,axes=plt.subplots(2)
ax1=axes[0]
ax2=axes[1]

for epoch in range(60):  # loop over the dataset multiple times
	
	model_registrar.to(device) # 放到设备上
	
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		#optimizer.zero_grad()
		optimizer_w.zero_grad()
		optimizer_b.zero_grad()

		# forward + backward + optimize
		
		loss1,acc_sum1 = model_1.train_loss(inputs, labels)
		loss2,acc_sum2 = model_2.train_loss(inputs, labels)
		loss = (loss1 + loss2) / 2
		acc_sum = (acc_sum1 + acc_sum2)/2
		loss.backward()
		#optimizer.step()
		optimizer_w.step()
		optimizer_b.step()

		# print statistics
		running_loss = loss.item()
		
		if (count+1) % 10 == 0:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f acc: %.4f' %
				  (epoch + 1, count + 1, running_loss, acc_sum))
				  
		if (count+1) % 5000 == 0:
			model_registrar.save_models(count)
		
		# 实时绘图 选取合适的时间间隔 
		
		ax1.plot(count,running_loss,'.b')
		ax2.plot(count,acc_sum,'.r')
		plt.draw()
		plt.pause(0.001)
		
		count = count + 1
			

print('Finished Training')
