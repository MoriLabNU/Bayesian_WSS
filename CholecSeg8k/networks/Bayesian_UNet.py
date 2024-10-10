"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_up1 = self.up1(x4, x3)
        x_up2 = self.up2(x_up1, x2)
        x_up3 = self.up3(x_up2, x1)
        x_up4 = self.up4(x_up3, x0)
        output = self.out_conv(x_up4)
        return output


class BDL_L_UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(BDL_L_UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': in_chns,
                   'bilinear': False,
                   'acti_func': 'relu'}
        self.encoder1 = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.fc_mu = nn.Linear(256 * 15 * 27, 256)
        self.fc_var = nn.Linear(256 * 15 * 27, 256)
        self.transform = nn.Linear(256, 256 * 15 * 27)
        self.sigmoid = nn.Sigmoid()

        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'bilinear': False,
                   'acti_func': 'relu'}

        params3 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 512],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'bilinear': False,
                   'acti_func': 'relu'}

        self.encoder2 = Encoder(params2)
        self.decoder2 = Decoder(params3)

    def reparameterize(self, mu, logvar, sample_time):

        B, D = mu.size()

        if sample_time == 1:

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).cuda()
            return eps * std + mu

        else:
            latent = None
            for i in range(sample_time):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std).cuda()
                lat = eps * std + mu
                if i == 0:
                    latent = lat.unsqueeze(0)
                else:
                    latent = torch.cat((latent, lat.unsqueeze(0)), 0)
            return latent.view(B * sample_time, D)

    def forward(self, x, sample_time):
        B, _, H, W = x.size()

        feature_encoder1 = self.encoder1(x)

        feature_flatten_encoder1 = torch.flatten(feature_encoder1[4], start_dim=1)

        mu = self.fc_mu(feature_flatten_encoder1)
        log_var = self.fc_var(feature_flatten_encoder1)

        z = self.reparameterize(mu, log_var, sample_time)
        z_ = self.transform(z).view(-1, 256, 15, 27) # 240/16=15, 432/16=27

        feature_encoder1[0] = feature_encoder1[0].repeat(sample_time, 1, 1, 1)
        feature_encoder1[1] = feature_encoder1[1].repeat(sample_time, 1, 1, 1)
        feature_encoder1[2] = feature_encoder1[2].repeat(sample_time, 1, 1, 1)
        feature_encoder1[3] = feature_encoder1[3].repeat(sample_time, 1, 1, 1)
        feature_encoder1[4] = z_

        x_gen = self.decoder1(feature_encoder1)
        x_gen = self.sigmoid(x_gen)
        x_gen = x_gen.view(sample_time, B, -1, H, W)

        feature_encoder2 = self.encoder2(x)
        feature_encoder2[0] = feature_encoder2[0].repeat(sample_time, 1, 1, 1)
        feature_encoder2[1] = feature_encoder2[1].repeat(sample_time, 1, 1, 1)
        feature_encoder2[2] = feature_encoder2[2].repeat(sample_time, 1, 1, 1)
        feature_encoder2[3] = feature_encoder2[3].repeat(sample_time, 1, 1, 1)
        feature_encoder2[4] = feature_encoder2[4].repeat(sample_time, 1, 1, 1)
        feature_encoder2[4] = torch.cat([z_, feature_encoder2[4]], dim=1)

        y = self.decoder2(feature_encoder2)
        y = y.view(sample_time, B, -1, H, W)

        return mu, log_var, x_gen, y


class BDL_MC_UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(BDL_MC_UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

