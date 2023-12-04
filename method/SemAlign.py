import torch.nn as nn
import torch


class SemAlign(nn.Module):
    def __init__(self, v_size, s_size, h_size=2048, drop=0.0):
        super(SemAlign, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(v_size + s_size, h_size),
            nn.LeakyReLU(0.2))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(h_size, v_size)

    def forward(self, semantic, contexts):
        input = torch.cat((semantic, contexts), -1)
        fusion = self.model(input)
        fusion = self.drop(fusion)
        fusion = self.fc(fusion)
        return fusion
