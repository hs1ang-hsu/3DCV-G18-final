import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange
from common.temporal_convolution import TemporalModel
from torch.autograd import Variable


class TransformNet(nn.Module):
    def __init__(
        self,
        kp=34,
        dim=3,
        hidden_dim=256
    ):
        super().__init__()
        
        # layers
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(hidden_dim)
        self.norm4 = nn.BatchNorm1d(hidden_dim//2)
        self.norm5 = nn.BatchNorm1d(hidden_dim//4)
        
        self.cnn1 = nn.Conv2d(1,  64, (1,dim))
        self.cnn2 = nn.Conv2d(64, 128, (1,1))
        self.cnn3 = nn.Conv2d(128, hidden_dim, (1,1))
        
        self.maxpool = nn.MaxPool2d((kp,1))
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc3 = nn.Linear(hidden_dim//4, dim*dim)
        
        # activation functions
        self.act_layer = nn.ReLU()
        
        # params
        self.K = dim
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # x.shape = (b*f, kp, 3)
        batch_size = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = x.permute((0,3,1,2)) # x.shape = (b*f, 1, kp, 3)
        x = self.act_layer(self.norm1(self.cnn1(x)))
        x = self.act_layer(self.norm2(self.cnn2(x)))
        x = self.act_layer(self.norm3(self.cnn3(x)))
        
        # x.shape = (b*f, 128, kp, 1)
        x = self.maxpool(x)
        x = x.view(-1, self.hidden_dim)
        x = self.act_layer(self.norm4(self.fc1(x)))
        x = self.act_layer(self.norm5(self.fc2(x)))
        x = self.fc3(x)
        
        B = Variable(torch.eye(self.K).flatten()).view(1,self.K*self.K).repeat(batch_size,1)
        if x.is_cuda:
            B = B.cuda()
        x = x + B
        return x.view(-1, self.K, self.K)

class EmotionClassifier(nn.Module):
    def __init__(
        self,
        kp=34,
        feature_dim=3,
        hidden_dim=256,
        channels=1024,
        out_dim=64,
        cls=7,
        trans=False,
    ):
        super().__init__()
        
        # layer
        if trans:
            self.feature_transform = TransformNet(kp, 3, hidden_dim)
        self.feature_spatial = nn.Linear(3, feature_dim)
        self.temporal_convolution = TemporalModel(kp, feature_dim, out_dim, [3,3,3], channels=channels)
        self.final = nn.Linear(out_dim, cls)
        
        # activation functions
        self.act_layer = nn.ReLU()
        
        # param
        self.trans = trans
        

    def forward(self, x, y):
        # x.shape = (batch, frames, keypoints, 3), keypoints=34
        b, f, p, _ = x.shape
        
        if self.trans:
            x = rearrange(x, 'b f p c -> (b f) p c', )
            T_input = self.feature_transform(x) # (b*f, p, feature_dim)
            x = torch.bmm(x, T_input)
            x = x.view(b, f, p, -1)
        
        x = self.feature_spatial(x)
        
        # temporal convolution
        x = self.temporal_convolution(x)
        x = self.final(self.act_layer(x))
        x = x.view(b, -1)
        
        y = torch.squeeze(y)
        return x.max(1)[1], F.cross_entropy(x, y.long())

if __name__ == '__main__':
    test_layer = EmotionClassifier()
    
    model_params = 0
    for parameter in test_layer.parameters():
        model_params += parameter.numel()
    print('Trainable parameter count:', model_params)
    
    A = torch.randn(8,27,34,3)
    B = torch.randn(8,1)
    test_layer(A)