import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.double_conv(x)
        return x

        
class AttentionBlock(nn.Module):
    '''
    stride = 2; please set --attention parameter to 1 when activate this block. the result name should include "att2".
    To fit in the structure of the V-net: F_l = F_int
    '''
    def __init__(self, F_g, F_l, F_int, scale_factor=1, mode="trilinear"):
        super(AttentionBlock, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),  # reduce num_channels
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=scale_factor, stride=scale_factor, padding=0, bias=True),  # downsize
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, visualize=False):
        '''
        :param g: gate signal from coarser scale
        :param x: the output of the l-th layer in the encoder
        :param visualize: enable this when plotting attention matrix
        :return:
        '''
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        relu = self.relu(g1 + x1)
        sig = self.psi(relu)
        alpha = nn.functional.interpolate(sig, scale_factor=self.scale_factor, mode=self.mode)

        if visualize:
            return alpha
        else:
            return x * alpha
            
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=1, stride=1),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Make sure size data when you combine are the same size
        diffX = x2.size()[4] - x1.size()[4]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[2] - x1.size()[2]
        
        x1 = torch.nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpP(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*3, out_channels)
    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        # Make sure size data when you combine are the same size
        diffX1 = x3.size()[4] - x1.size()[4]
        diffY1 = x3.size()[3] - x1.size()[3]
        diffZ1 = x3.size()[2] - x1.size()[2]
        
        diffX2 = x3.size()[4] - x2.size()[4]
        diffY2 = x3.size()[3] - x2.size()[3]
        diffZ2 = x3.size()[2] - x2.size()[2]
        
        x2 = torch.nn.functional.pad(x2, (diffX2 // 2, diffX2 - diffX2 // 2, diffY2 // 2, diffY2 - diffY2 // 2, diffZ2 // 2, diffZ2 - diffZ2 // 2))
        x1 = torch.nn.functional.pad(x1, (diffX1 // 2, diffX1 - diffX1 // 2, diffY1 // 2, diffY1 - diffY1 // 2, diffZ1 // 2, diffZ1 - diffZ1 // 2))
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)

class UpPP(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*4, out_channels)
    def forward(self, x1, x2, x3, x4):
        x1 = self.up(x1)
        # Make sure size data when you combine are the same size
        diffX1 = x4.size()[4] - x1.size()[4]
        diffY1 = x4.size()[3] - x1.size()[3]
        diffZ1 = x4.size()[2] - x1.size()[2]
        
        diffX2 = x4.size()[4] - x2.size()[4]
        diffY2 = x4.size()[3] - x2.size()[3]
        diffZ2 = x4.size()[2] - x2.size()[2]
        
        diffX3 = x4.size()[4] - x3.size()[4]
        diffY3 = x4.size()[3] - x3.size()[3]
        diffZ3 = x4.size()[2] - x3.size()[2]
        
        x3 = torch.nn.functional.pad(x3, (diffX3 // 2, diffX3 - diffX3 // 2, diffY3 // 2, diffY3 - diffY3 // 2, diffZ3 // 2, diffZ3 - diffZ3 // 2))
        x2 = torch.nn.functional.pad(x2, (diffX2 // 2, diffX2 - diffX2 // 2, diffY2 // 2, diffY2 - diffY2 // 2, diffZ2 // 2, diffZ2 - diffZ2 // 2))
        x1 = torch.nn.functional.pad(x1, (diffX1 // 2, diffX1 - diffX1 // 2, diffY1 // 2, diffY1 - diffY1 // 2, diffZ1 // 2, diffZ1 - diffZ1 // 2))
        x = torch.cat([x4, x3, x2, x1], dim=1)
        return self.conv(x)   

class UpPPP(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*5, out_channels)
    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.up(x1)
        # Make sure size data when you combine are the same size
        diffX1 = x5.size()[4] - x1.size()[4]
        diffY1 = x5.size()[3] - x1.size()[3]
        diffZ1 = x5.size()[2] - x1.size()[2]
        
        diffX2 = x5.size()[4] - x2.size()[4]
        diffY2 = x5.size()[3] - x2.size()[3]
        diffZ2 = x5.size()[2] - x2.size()[2]
        
        diffX3 = x5.size()[4] - x3.size()[4]
        diffY3 = x5.size()[3] - x3.size()[3]
        diffZ3 = x5.size()[2] - x3.size()[2]
        
        diffX4 = x5.size()[4] - x4.size()[4]
        diffY4 = x5.size()[3] - x4.size()[3]
        diffZ4 = x5.size()[2] - x4.size()[2]
        
        x4 = torch.nn.functional.pad(x4, (diffX4 // 2, diffX4 - diffX4 // 2, diffY4 // 2, diffY4 - diffY4 // 2, diffZ4 // 2, diffZ4 - diffZ4 // 2))        
        x3 = torch.nn.functional.pad(x3, (diffX3 // 2, diffX3 - diffX3 // 2, diffY3 // 2, diffY3 - diffY3 // 2, diffZ3 // 2, diffZ3 - diffZ3 // 2))
        x2 = torch.nn.functional.pad(x2, (diffX2 // 2, diffX2 - diffX2 // 2, diffY2 // 2, diffY2 - diffY2 // 2, diffZ2 // 2, diffZ2 - diffZ2 // 2))
        x1 = torch.nn.functional.pad(x1, (diffX1 // 2, diffX1 - diffX1 // 2, diffY1 // 2, diffY1 - diffY1 // 2, diffZ1 // 2, diffZ1 - diffZ1 // 2))
        x = torch.cat([x5, x4, x3, x2, x1], dim=1)
        return self.conv(x)        


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.out_conv(x)        

