from .model_blocks import *

class UNET3DPP(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        
        self.x00 = DoubleConv(self.in_channels, self.out_channels)
        # UNet3D ++ L1
        self.down_to_x10 = Down(self.out_channels, self.out_channels*2)
        self.up_to_x01 = Up(self.out_channels*2, self.out_channels)
        # UNet3D ++ L2
        self.down_to_x20 = Down(self.out_channels*2, self.out_channels*4)
        self.up_to_x11 = Up(self.out_channels*4, self.out_channels*2)
        self.up_to_x02 = UpP(self.out_channels*2, self.out_channels)
        # UNet3D ++ L3
        self.down_to_x30 = Down(self.out_channels*4, self.out_channels*8)
        self.up_to_x21 = Up(self.out_channels*8, self.out_channels*4)
        self.up_to_x12 = UpP(self.out_channels*4, self.out_channels*2)
        self.up_to_x03 = UpPP(self.out_channels*2, self.out_channels)
        # UNet3D ++ L4
        self.down_to_x40 = Down(self.out_channels*8, self.out_channels*16)
        self.up_to_x31 = Up(self.out_channels*16, self.out_channels*8)
        self.up_to_x22 = UpP(self.out_channels*8, self.out_channels*4)
        self.up_to_x13 = UpPP(self.out_channels*4, self.out_channels*2)
        self.up_to_x04 = UpPPP(self.out_channels*2, self.out_channels)
        # Attention Blocks
        self.ag_0 = AttentionBlock(512, 256, 256)
        self.ag_2 = AttentionBlock(256, 128, 128)  # forward(g, x)
        self.ag_1 = AttentionBlock(128, 64, 64)
        self.ag_ = AttentionBlock(64, 32, 32)

        # output
        self.out = Out(self.out_channels, self.n_classes)
        
        self.dropout = nn.Dropout3d(0.25)
    def forward(self, x):
        x00 = self.x00(x)
        # UNet3D ++ L1
        x10 = self.down_to_x10(x00)
        x01 = self.up_to_x01(x10, x00)
        atten = self.ag_(x10,x01)
        x01 = self.dropout(x0)
        #print(x01.shape)

        # UNet3D ++ L2
        x20 = self.down_to_x20(x10)
        x11 = self.up_to_x11(x20, x10)
        print(x11.shape)
        x11 = self.dropout(x11)
        x02 = self.up_to_x02(x11, x01, x00)
        x02 = self.dropout(x02)
        print(x20.shape)
        # UNet3D ++ L3
        x30 = self.down_to_x30(x20)
        x21 = self.up_to_x21(x30, x20)
        x21 = self.dropout(x21)
        x12 = self.up_to_x12(x21, x11, x10)
        x12 = self.dropout(x12)
        x03 = self.up_to_x03(x12, x02, x01, x00)
        x03 = self.dropout(x03)
        print(x30.shape)

        # UNet3D ++ L4
        x40 = self.down_to_x40(x30)
        x31 = self.up_to_x31(x40, x30)
        x31 = self.dropout(x31)
        x22 = self.up_to_x22(x31, x21, x20)
        x22 = self.dropout(x22)
        x13 = self.up_to_x13(x22, x12, x11, x10)
        x13 = self.dropout(x13)
        x04 = self.up_to_x04(x13, x03, x02, x01, x00)
        x04 = self.dropout(x04)
        print(x04.shape)

        
        # Output
        out = self.out(x04)
        print(out.shape)
        return out