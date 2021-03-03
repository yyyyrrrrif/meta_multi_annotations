import torch
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, n_class):
        super(net, self).__init__()
        self.fc1 = nn.Linear(4*4*512, 128)
        self.fc2 = nn.Linear(128, n_class)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = x.view(-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = F.softmax(out)
        return out


