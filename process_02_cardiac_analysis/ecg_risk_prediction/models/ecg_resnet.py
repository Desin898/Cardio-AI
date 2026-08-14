import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MultiBranch1DResNet34(nn.Module):
    """
    Multi-Branch 1D-ResNet34 backbone for 12-lead ECG signal classification.
    Inputs:
        x: Tensor of shape (B, 12, N) or (B, 1, 12, N)
    Outputs:
        logits: Tensor of shape (B, num_classes)
    """
    def __init__(self, num_classes=4, in_channels=12):
        super(MultiBranch1DResNet34, self).__init__()
        self.in_channels = in_channels

        # Lead-branch feature extractors: 1D convolution per lead channel
        self.lead_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU()
            ) for _ in range(in_channels)
        ])

        # Fused backbone after concatenating 12 lead branches (16 * 12 = 192 channels)
        self.conv1 = nn.Conv1d(192, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.in_planes = 64
        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Handle 4D inputs (B, 1, 12, N) -> (B, 12, N)
        if x.dim() == 4:
            if x.size(1) == 1 and x.size(2) == 12:
                x = x.squeeze(1)
            elif x.size(1) == 12 and x.size(2) == 1:
                x = x.squeeze(2)

        # Multi-branch lead processing
        branch_outputs = []
        for i in range(self.in_channels):
            lead_signal = x[:, i:i+1, :]  # Shape: (B, 1, N)
            branch_outputs.append(self.lead_branches[i](lead_signal))

        # Concatenate features along channel dimension: (B, 16 * 12 = 192, N)
        fused = torch.cat(branch_outputs, dim=1)

        out = F.relu(self.bn1(self.conv1(fused)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)
