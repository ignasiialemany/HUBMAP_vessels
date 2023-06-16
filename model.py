
import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        
        super(UNET,self).__init__()
        
        #We will store all convolutions in a list for up and down paths
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        #Middle part 
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        
        #Up part of UNET
        for feature in features[::-1]:
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature ))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        #Here we will deal with the network where x is the "tensor"
        store_features = []
        out = x
        for down in self.downs:
            out = down(out)
            store_features.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)
        store_features = store_features[::-1]
        
        for i in range (0,len(self.ups),2):
            #up sample
            out = self.ups[i](out)
            
            #If the initial image is not divisible by 16 we need to resize the out shape 
            if out.shape != store_features[i//2].shape:
                out = TF.resize(out, size=store_features[i//2].shape[2:])
            
            #concatenate with stored_features
            concatenated =  torch.cat((store_features[i//2],out),dim=1)
            #double convolute concatenated
            out = self.ups[i+1](concatenated)
        
        return self.final_conv(out)

            #concatenate bottom and store_features[i//2]
def test():
    x = torch.randn((3,1,161,161))
    model = UNET(in_channels=1, out_channels = 1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()