#%%
import torch
x = torch.arange(12)
y = torch.tensor([1,4,5,6])
print(x,y)

import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')

with open(data_file,'w' ) as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd
data = pd.read_csv(data_file)
print(data)
#%% 处理缺失值
inputs , outputs = data.iloc[:,0:2], data.iloc[:,2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
# 将nan看作一个值，将Alley属性转为one-hot编码格式
inputs = pd.get_dummies(inputs,dummy_na=True)
print(inputs)
#%%
X , y = torch.tensor(inputs.values),torch.tensor(outputs.values)
print(X,y)

#%%
A = torch.arange(20).reshape(5,4)
B = A[:]
# B = A.clone()
# print(id(B) == id(A))
print(A * B)

#%%
x = torch.arange(4,dtype=torch.float32)
print(x,x.sum())

print('A:',A)
# 把0这维去掉，[5,4]剩下[4]
print('A.sum(axis=0):',A.sum(axis=0))
B = A.reshape(2,2,5)
print('B:',B)
print('B.sum(axis=[0,1]):',B.sum(axis=[0,1]))
# print(A.float().mean(axis=0))
print(A.float().mean(dim=0))
#%%
sum_A = A.sum(axis=1,keepdim=True)
print(sum_A)
print(A / sum_A)
# 沿着某个轴计算A元素的累计总和
print(A.cumsum(axis=0))
