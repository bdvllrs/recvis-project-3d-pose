import torch
from torchvision.models import resnet34
import torch.nn as nn


class Resnet2DModel(nn.Module):
    def __init__(self, n_joints, n_frames=1):
        super(Resnet2DModel, self).__init__()
        self.resnet = resnet34(pretrained=True)
        model_conv = nn.Sequential(*list(self.resnet.children())[:-3])
        for param in model_conv.parameters():
            param.requires_grad = False
        fc_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(fc_features, n_joints * 64 * 64)

        self.n_frames = n_frames
        self.n_joints = n_joints

        self.fc = nn.Sequential(
            nn.Linear(n_joints * 64 * 64 * n_frames, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_joints * 2),
            nn.ReLU()
        )

    def forward(self, inputs):
        out = torch.zeros(inputs.size(0), self.n_frames, self.n_joints * 64 * 64).to(inputs.get_device())
        for k in range(inputs.size(1)):
            o = self.resnet(inputs[:, k])
            out[:, k] = o
        out = out.view(out.size(0), -1)
        return self.fc(out)
