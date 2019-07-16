''' Pytorch implementation of HomographyNet.
    Reference: https://arxiv.org/pdf/1606.03798.pdf and https://github.com/mazenmel/Deep-homography-estimation-Pytorch
    Currently supports translations (2 params)
    The network reads pair of images (tensor x: [B,2*C,W,H])
    and outputs parametric transformations (tensor out: [B,n_params]).'''

import torch
import torch.nn as nn
import lanczos


class ShiftNet(nn.Module):
    ''' ShiftNet, a neural network for sub-pixel registration and interpolation with lanczos kernel. '''
    
    def __init__(self, in_channel=1):
        '''
        Args:
            in_channel : int, number of input channels
        '''
        
        super(ShiftNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(2 * in_channel, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2, bias=False)
        self.fc2.weight.data.zero_() # init the weights with the identity transformation

    def forward(self, x):
        '''
        Registers pairs of images with sub-pixel shifts.
        Args:
            x : tensor (B, 2*C_in, H, W), input pairs of images
        Returns:
            out: tensor (B, 2), translation params
        '''

        x[:, 0] = x[:, 0] - torch.mean(x[:, 0], dim=(1, 2)).view(-1, 1, 1)
        x[:, 1] = x[:, 1] - torch.mean(x[:, 1], dim=(1, 2)).view(-1, 1, 1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = out.view(-1, 128 * 16 * 16)
        out = self.drop1(out)  # dropout on spatial tensor (C*W*H)

        out = self.fc1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        return out

    def transform(self, theta, I, device="cpu"):
        '''
        Shifts images I by theta with Lanczos interpolation.
        Args:
            theta : tensor (B, 2), translation params
            I : tensor (B, C_in, H, W), input images
        Returns:
            out: tensor (B, C_in, W, H), shifted images
        '''

        self.theta = theta
        new_I = lanczos.lanczos_shift(img=I.transpose(0, 1),
                                      shift=self.theta.flip(-1),  # (dx, dy) from register_batch -> flip
                                      a=3, p=5)[:, None]
        return new_I