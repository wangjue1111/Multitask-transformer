import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self,max_len,num_layers,kernel_size):
        super(TextCNN, self).__init__()
        self.conv1ds=nn.ModuleList([nn.Conv1d(max_len,max_len,kernel_size,1,padding=kernel_size-2) for i in range(num_layers)])

    def forward(self,x):

        for conv1d in self.conv1ds:
            x=x+F.tanh(conv1d(x))
        x=F.avg_pool1d(x,kernel_size=x.shape[-1])
        return x.squeeze(-1)



