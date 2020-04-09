
import torch
import torch.nn as nn


class Rezero_Linear(nn.Module):
	def __init__(self,inputs,outputs):
		super(Rezero_Linear,self).__init__()
		self.inputs = inputs
		self.outputs = outputs
		self.resweight = nn.Parameter(torch.Tensor([0]))
		self.fc = nn.Linear(self.inputs,self.outputs)
		
		self.init_weights() # 初始化权重
	
	def init_weights(self):
		torch.nn.init.xavier_normal_(self.fc.weight, 
						gain=torch.sqrt(torch.tensor(2.)))
						
	def forward(self,x):
		x = x + self.resweight * torch.relu(self.fc(x))
		return x

"""
a = torch.randn(32, 512)
model = Rezero_Linear(512,512)
ou = model(a)
print(ou.shape)
"""

