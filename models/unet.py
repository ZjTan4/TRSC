import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.glcm import get_glcm, get_glcm_features, get_glcm_features1

class Block(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(), 
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), 
            nn.BatchNorm2d(out_ch),
            nn.ReLU(), 
        )
    
    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, chs) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)
        ])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        '''
        torch.Size([10, 128, 69, 86])
        torch.Size([10, 64, 138, 172])
        torch.Size([10, 32, 276, 344])
        torch.Size([10, 16, 552, 688])
                    10,  3, 552, 688
        '''
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, chs) -> None:
        super().__init__()
        self.upconvolutions = nn.ModuleList([
            nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)
        ])
        self.decoder_blocks = nn.ModuleList([
            Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)
        ])
    
    def forward(self, x, encoder_features):
        '''
        torch.Size([10, 128, 69, 86])
        torch.Size([10, 64, 138, 172])
        torch.Size([10, 32, 276, 344])
        torch.Size([10, 16, 552, 688])
                    10,  6, 552, 688
        '''
        for i in range(len(self.decoder_blocks)):
            x = self.upconvolutions[i](x)
            encoder_feature = self.crop(x, encoder_features[i])
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decoder_blocks[i](x)
        return x

    def crop(self, x, encoder_features):
        _, _, H, W = x.shape
        encoder_feature = torchvision.transforms.CenterCrop([H, W])(encoder_features)
        return encoder_feature

class UNet(nn.Module):
    def __init__(self, 
        encoder_chs=(3, 16, 32, 64, 128), 
        decoder_chs=(128, 64, 32, 16), 
        num_class=1, 
    ) -> None:
        super().__init__()
        self.encoder = Encoder(encoder_chs)
        self.decoder = Decoder(decoder_chs)
        self.head = nn.Conv2d(decoder_chs[-1], num_class, 1)

    def forward(self, x):
        encoder_features = self.encoder(x)[::-1]
        # for i in range(len(encoder_features)):
        #     print(encoder_features[i].shape)
        # exit(0)
        out = self.decoder(encoder_features[0], encoder_features[1:])
        out = self.head(out)
        return F.log_softmax(out, dim=1)

class UNet_GLCM(nn.Module):
    def __init__(self, 
        encoder_chs=(12, 16, 32, 64, 128), 
        decoder_chs=(128, 64, 32, 16), 
        num_class=1, 
        device='cuda:9'
    ) -> None:
        super().__init__()
        self.encoder = Encoder(encoder_chs)
        self.decoder = Decoder(decoder_chs)
        self.head = nn.Conv2d(decoder_chs[-1], num_class, 1)
        self.device = device

    def forward(self, x):
        batch, c, h, w = x.shape
        glcm_weight = 0.5
        glcm_features = torch.zeros((batch, 9, h, w))
        # glcm_features = torch.zeros((batch, 1, h, w))
        for i in range(batch):
            gray_img = x[i, :, :, :].mean(dim=0)
            glcm = get_glcm(gray_img, device=self.device)
            glcm_features[i, :, :, :] = get_glcm_features(glcm)
            # glcm_features[i, :, :, :] = get_glcm_features1(glcm)
        glcm_features = glcm_features * glcm_weight
        glcm_features = glcm_features.to(self.device)
        x = torch.cat((x, glcm_features), dim=1)
        encoder_features = self.encoder(x)[::-1]
        out = self.decoder(encoder_features[0], encoder_features[1:])
        out = self.head(out)
        return F.log_softmax(out, dim=1)