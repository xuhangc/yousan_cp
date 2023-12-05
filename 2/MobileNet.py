import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        # Initialize the layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5)
        
        self.conv2_1_dw = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.bn2_1_dw = nn.BatchNorm2d(32, eps=1e-5)
        
        self.conv2_1_sep = nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False)
        self.bn2_1_sep = nn.BatchNorm2d(64, eps=1e-5)
        
        self.conv2_2_dw = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.bn2_2_dw = nn.BatchNorm2d(64, eps=1e-5)
        
        self.conv2_2_sep = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
        self.bn2_2_sep = nn.BatchNorm2d(128, eps=1e-5)
        
        self.conv3_1_dw = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.bn3_1_dw = nn.BatchNorm2d(128, eps=1e-5)
        
        self.conv3_1_sep = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.bn3_1_sep = nn.BatchNorm2d(128, eps=1e-5)
        
        self.conv3_2_dw = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=False)
        self.bn3_2_dw = nn.BatchNorm2d(128, eps=1e-5)
        
        self.conv3_2_sep = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.bn3_2_sep = nn.BatchNorm2d(256, eps=1e-5)
        
        self.conv4_1_dw = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False)
        self.bn4_1_dw = nn.BatchNorm2d(256, eps=1e-5)
        
        self.conv4_1_sep = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.bn4_1_sep = nn.BatchNorm2d(256, eps=1e-5)
        
        self.conv4_2_dw = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.bn4_2_dw = nn.BatchNorm2d(256, eps=1e-5)
        
        self.conv4_2_sep = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        self.bn4_2_sep = nn.BatchNorm2d(512, eps=1e-5)
        
        self.pool6 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc7_mouth = nn.Conv2d(512, 2, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2_1_dw(x)
        x = self.bn2_1_dw(x)
        x = F.relu(x)
        
        x = self.conv2_1_sep(x)
        x = self.bn2_1_sep(x)
        x = F.relu(x)
        
        x = self.conv2_2_dw(x)
        x = self.bn2_2_dw(x)
        x = F.relu(x)
        
        x = self.conv2_2_sep(x)
        x = self.bn2_2_sep(x)
        x = F.relu(x)
        
        x = self.conv3_1_dw(x)
        x = self.bn3_1_dw(x)
        x = F.relu(x)
        
        x = self.conv3_1_sep(x)
        x = self.bn3_1_sep(x)
        x = F.relu(x)
        
        x = self.conv3_2_dw(x)
        x = self.bn3_2_dw(x)
        x = F.relu(x)
        
        x = self.conv3_2_sep(x)
        x = self.bn3_2_sep(x)
        x = F.relu(x)
        
        x = self.conv4_1_dw(x)
        x = self.bn4_1_dw(x)
        x = F.relu(x)
        
        x = self.conv4_1_sep(x)
        x = self.bn4_1_sep(x)
        x = F.relu(x)
        
        x = self.conv4_2_dw(x)
        x = self.bn4_2_dw(x)
        x = F.relu(x)
        
        x = self.conv4_2_sep(x)
        x = self.bn4_2_sep(x)
        x = F.relu(x)
        
        x = self.pool6(x)
        x = self.fc7_mouth(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the loss calculation
        
        return x

    def _initialize_weights(self):
        # Initialize weights using the MSRA filler which is equivalent to kaiming_normal_ in PyTorch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    t = torch.randn(1, 3, 160, 160).cuda()
    model = MobileNet().cuda()
    res = model(t)
    print(res.shape)