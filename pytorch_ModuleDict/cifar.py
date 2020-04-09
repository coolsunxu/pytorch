

import torch.nn as nn
import torch.nn.functional as F

class CIFAR(object):
	def __init__(self, model_registrar, device):
					 
		# 一些基本信息的设置
		self.model_registrar = model_registrar
		self.device = device
		self.node_modules = dict() # 字典
		self.criterion = nn.CrossEntropyLoss()
		

	def add_submodule(self, name, model_if_absent):
		# 注册模型 同时放到设备上
		self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)


	def clear_submodules(self):
		self.node_modules.clear() # 清除字典


	def create_graphical_model(self):
		self.clear_submodules()
		self.add_submodule('conv1', model_if_absent=nn.Conv2d(3, 6, 5))
		self.add_submodule('pool', model_if_absent=nn.MaxPool2d(2, 2))
		self.add_submodule('conv2', model_if_absent=nn.Conv2d(6, 16, 5))
		
		self.add_submodule('fc1', model_if_absent=nn.Linear(16 * 5 * 5, 120))
		self.add_submodule('fc2', model_if_absent=nn.Linear(120, 84))
		self.add_submodule('fc3', model_if_absent=nn.Linear(84, 10))
		
	def encoder(self, x):
		
		x = self.node_modules['pool'](F.relu(self.node_modules['conv1'](x)))
		x = self.node_modules['pool'](F.relu(self.node_modules['conv2'](x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.node_modules['fc1'](x))
		x = F.relu(self.node_modules['fc2'](x))
		x = self.node_modules['fc3'](x)

		return x

	def train_loss(self, inputs, labels):
		
		outputs = self.encoder(inputs)
		loss = self.criterion(outputs, labels)
		acc_sum = (outputs.argmax(dim=1) == labels).float().sum().cpu().item()/12
		return loss,acc_sum


   
