
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
	def __init__(self,fin,fout=1):
		super(TemporalAttention,self).__init__()
		self.fin = fin # 输入维度
		self.fout = fout # 输出维度 这里为1 求得是分数
		
		# 自定义可学习参数
		self.w = nn.Parameter(torch.Tensor(self.fin, self.fout))
		# 初始化自定义参数
		nn.init.xavier_uniform_(self.w, gain=1.414)
		
	def forward(self,h): # h:[bs,seq,fin]
		x = h # [bs,seq,fin]
		alpha = torch.matmul(h,self.w) # [bs,seq,1] fout==1
		alpha = F.softmax(torch.tanh(alpha),1) # [bs,seq,1]
		x = torch.einsum('ijk,ijm->ikm', alpha, x) # [bs,1,fin]
		#x = torch.matmul(alpha.permute(0,2,1),x) # [bs,1,fin]
		return torch.squeeze(x,1)
		
a = torch.randn(42,8,64)
model = TemporalAttention(64,1)
z = model(a)
print(z.shape)

