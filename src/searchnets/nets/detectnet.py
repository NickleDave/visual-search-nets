import torch
import torch.nn as nn


class DetectNet(nn.Module):
    def __init__(self,
                 vis_sys,
                 num_classes,
                 vis_sys_n_out,
                 embedding_n_out=512):
        super(DetectNet, self).__init__()
        self.vis_sys = vis_sys
        self.embedding = nn.Sequential(nn.Linear(in_features=num_classes,
                                                 out_features=embedding_n_out),
                                       nn.ReLU(inplace=True),
                                       )
        self.decoder = nn.Linear(in_features=vis_sys_n_out + embedding_n_out,
                                 out_features=1)  # always 1, because it indicates target present or absent

    def forward(self, img, query):
        vis_out = self.vis_sys(img)
        query_out = self.embedding(query)
        out = self.decoder(
            torch.cat((vis_out, query_out), dim=1)
        )
        return out
