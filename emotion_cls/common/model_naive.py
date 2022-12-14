import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange

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
        self.fc1 = nn.Linear(3*kp, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        
        self.cnn = torch.nn.Conv1d(in_channels=27, out_channels=1, kernel_size=1)
        
        self.fc4 = nn.Linear(1024, 256)
        self.fc5 = nn.Linear(256, 7)
        
        self.drop = nn.Dropout(0.5)
        
        # activation functions
        self.act_layer = nn.ReLU()
        

    def forward(self, x, y):
        # x.shape = (batch, frames, keypoints, 3), keypoints=34
        b, f, p, _ = x.shape
        
        x = rearrange(x, 'b f p c -> b f (p c)', )
        x = self.drop(self.act_layer(self.fc1(x)))
        x = self.drop(self.act_layer(self.fc2(x)))
        x = self.drop(self.act_layer(self.fc3(x)))
        
        x = self.cnn(x)
        x = self.drop(self.act_layer(self.fc4(x)))
        x = self.act_layer(self.fc5(x))
        
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