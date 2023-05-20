from __future__ import print_function

import torch.nn as nn

    
class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=1536,
                 out_features=768,
                 act_layer=nn.ReLU,
                 drop=0.):
        super().__init__()
        out_features = 768
        hidden_features = in_features * 2 
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        # x = self.drop(x)
        return x
    
class Linear(nn.Module):
    def __init__(self,
                 in_features,            
                 out_features=768):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)


    def forward(self, x):
        x = self.fc(x)
        return x



