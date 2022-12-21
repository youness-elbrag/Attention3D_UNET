from .model_blocks import *

class UNET3DPP(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        
        self.x00 = DoubleConv(self.in_channels, self.out_channels) # 4x64 ===> (64,128,128)
        # UNet3D ++ L1 
        self.down_to_x10 = Down(self.out_channels, self.out_channels*2) # 64x128
        self.up_to_x01 = Up(self.out_channels*2, self.out_channels) ## 128x64
        # UNet3D ++ L2
        self.down_to_x20 = Down(self.out_channels*2, self.out_channels*4)#128x256
        self.up_to_x11 = Up(self.out_channels*4, self.out_channels*2) ## (256x128)
        self.up_to_x02 = UpP(self.out_channels*2, self.out_channels)##(128x64)
        # UNet3D ++ L3
        self.down_to_x30 = Down(self.out_channels*4, self.out_channels*8)##(256x512)
        self.up_to_x21 = Up(self.out_channels*8, self.out_channels*4)##(512x256)
        self.up_to_x12 = UpP(self.out_channels*4, self.out_channels*2)##(256x128)
        self.up_to_x03 = UpPP(self.out_channels*2, self.out_channels)##(128x64)
        # UNet3D ++ L4
        self.down_to_x40 = Down(self.out_channels*8, self.out_channels*16)##(512x1024)
        self.up_to_x31 = Up(self.out_channels*16, self.out_channels*8)##(1024x512)
        self.up_to_x22 = UpP(self.out_channels*8, self.out_channels*4)##(512x256)
        self.up_to_x13 = UpPP(self.out_channels*4, self.out_channels*2)###(256x128)
        self.up_to_x04 = UpPPP(self.out_channels*2, self.out_channels)##(128x24)
        # Attention Blocks
        self.ag_3 = AttentionBlock(512, 256, 512)
        self.ag_2 = AttentionBlock(256, 128, 256)  # forward(g, x)
        self.ag_1 = AttentionBlock(128, 64, 128)
        self.ag_0 = AttentionBlock(64, 32,64)


        # output
        self.out = Out(self.out_channels, self.n_classes)##(64x3)
        
        self.dropout = nn.Dropout3d(0.25)
    def forward(self, x):
        x00 = self.x00(x)
        # UNet3D ++ L1
        x10 = self.down_to_x10(x00)
        ag_x00 =  self.ag_0(x10,x00) # ag_x00==> ag(x00)
        x01 = self.up_to_x01(x10,ag_x00) 
        x01 = self.dropout(x01)
        
        # UNet3D ++ L2
        x20 = self.down_to_x20(x10)
        ag_x20 = self.ag_1(x20,x10)   # ag_x20 ==> ag(x20)
        x11 = self.up_to_x11(x20, ag_x20) 
        x11 = self.dropout(x11)
        ag_x01 = self.ag_0(x11,x01)  ## ag_x01 ==> ag(x01)
        x02 = self.up_to_x02(x11, ag_x01, ag_x00)
        x02 = self.dropout(x02)
        
        # UNet3D ++ L3
        x30 = self.down_to_x30(x20)
        ag_x30 =  self.ag_2(x30,x20) ## ag_x30 ==> ag(x30)
        x21 = self.up_to_x21(x30, ag_x30)
        x21 = self.dropout(x21)
        x12 = self.up_to_x12(x21, x11, x10)
        x12 = self.dropout(x12)
        ag_x02 = self.ag_0(x12,x02) ## ag_x02 ==> ag(x02)
        x03 = self.up_to_x03(x12, ag_x02, ag_x01, ag_x00)
        x03 = self.dropout(x03)

        # UNet3D ++ L4
        x40 = self.down_to_x40(x30)
        ag_x40 =  self.ag_3(x40,x30) ## ag_x40 ==> ag(x40)
        x31 = self.up_to_x31(x40, ag_x40)
        x31 = self.dropout(x31)
        x22 = self.up_to_x22(x31, x21, x20)
        x22 = self.dropout(x22)
        x13 = self.up_to_x13(x22, x12, x11, x10)
        x13 = self.dropout(x13)
        ag_x03 = self.ag_0(x13,x03) ### ag_x03 ==> ag(x03)
        x04 = self.up_to_x04(x13, ag_x03, ag_x02, ag_x01, ag_x00)
        x04 = self.dropout(x04)

        
        # Output
        out = self.out(x04)
        return out