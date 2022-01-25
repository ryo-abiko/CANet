'''
Model of CANet
'''
import torch.nn as nn
import torch


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)

        return x


class colorBlock(nn.Module):
    def __init__(self, in_channels, act_func=nn.Mish(inplace=True)):
        super().__init__()

        self.spatialsqueze = nn.Sequential(
            nn.Linear(in_channels, 3, bias=True),
            nn.Sigmoid()
        )

        self.toColorLayer = nn.Conv2d(in_channels, 3, 3, padding=1)

    def forward(self, x):

        ccenhance = self.spatialsqueze(x.view(x.shape[0], x.shape[1], -1).mean(dim=2))
        output = self.toColorLayer(x)

        return torch.mul(output, ccenhance.view(x.shape[0], 3, 1, 1))


class scSEBlock(nn.Module):
    def __init__(self, in_channels, act_func=nn.Mish(inplace=True)):
        super().__init__()

        self.chanelsqueze = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

        self.spatialsqueze = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2, bias=True),
            act_func,
            nn.Linear(in_channels // 2, in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):

        sSEblock = self.chanelsqueze(x).view(x.shape[0], 1, x.shape[2], x.shape[3])
        sSEblock = torch.mul(x, sSEblock)

        cSEblock = self.spatialsqueze(x.view(x.shape[0], x.shape[1], -1).mean(dim=2))
        cSEblock = torch.mul(x, cSEblock.view(x.shape[0], x.shape[1], 1, 1))

        return sSEblock + cSEblock


class GCVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_func=nn.Mish(inplace=True)):
        super().__init__()

        self.upchannel = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            act_func
        )

        self.model = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_func,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_func
        )

        self.scse = scSEBlock(out_channels)

    def forward(self, x):

        uclayer = self.upchannel(x)

        return uclayer + self.scse(self.model(uclayer))


class GeneratorUNet_d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, n_residual_blocks=16):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = Interpolate(scale_factor=2, mode='bilinear')

        self.conv0_0 = GCVGGBlock(in_channels, nb_filter[0])
        self.conv1_0 = GCVGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = GCVGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = GCVGGBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = GCVGGBlock(nb_filter[3], nb_filter[4])
        self.conv5_0 = GCVGGBlock(nb_filter[4], nb_filter[5])

        self.conv0_1 = GCVGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = GCVGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = GCVGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = GCVGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv4_1 = GCVGGBlock(nb_filter[4] + nb_filter[5], nb_filter[4])

        self.conv0_2 = GCVGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = GCVGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = GCVGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        self.conv3_2 = GCVGGBlock(nb_filter[3] * 2 + nb_filter[4], nb_filter[3])

        self.conv0_3 = GCVGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = GCVGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        self.conv2_3 = GCVGGBlock(nb_filter[2] * 3 + nb_filter[3], nb_filter[2])

        self.conv0_4 = GCVGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.conv1_4 = GCVGGBlock(nb_filter[1] * 4 + nb_filter[2], nb_filter[1])

        self.conv0_5 = GCVGGBlock(nb_filter[0] * 5 + nb_filter[1], nb_filter[0])

        self.final4 = nn.Sequential(
            nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1),
            nn.BatchNorm2d(nb_filter[0]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(nb_filter[0], out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        return self.final4(x0_5)


class Discriminator_d(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.Mish(inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.Mish(inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class GeneratorUNet_r(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, n_residual_blocks=16):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = Interpolate(scale_factor=2, mode='bilinear')

        self.conv0_0 = GCVGGBlock(in_channels, nb_filter[0])
        self.conv1_0 = GCVGGBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = GCVGGBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = GCVGGBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = GCVGGBlock(nb_filter[3], nb_filter[4])
        self.conv5_0 = GCVGGBlock(nb_filter[4], nb_filter[5])

        self.conv0_1 = GCVGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = GCVGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = GCVGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = GCVGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv4_1 = GCVGGBlock(nb_filter[4] + nb_filter[5], nb_filter[4])

        self.conv0_2 = GCVGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = GCVGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = GCVGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        self.conv3_2 = GCVGGBlock(nb_filter[3] * 2 + nb_filter[4], nb_filter[3])

        self.conv0_3 = GCVGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = GCVGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        self.conv2_3 = GCVGGBlock(nb_filter[2] * 3 + nb_filter[3], nb_filter[2])

        self.conv0_4 = GCVGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.conv1_4 = GCVGGBlock(nb_filter[1] * 4 + nb_filter[2], nb_filter[1])

        self.conv0_5 = GCVGGBlock(nb_filter[0] * 5 + nb_filter[1], nb_filter[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1),
            nn.Mish(inplace=True),
        )
        self.final2 = scSEBlock(nb_filter[0])
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=3, padding=1),
            nn.Mish(inplace=True),
        )
        self.final4 = scSEBlock(nb_filter[0])

        self.final_channel_mod = colorBlock(nb_filter[0])

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        output = self.final4(self.final3(self.final2(self.final1(x0_5))))
        final_output = x[:, 0:3, :, :] + self.final_channel_mod(output)

        return final_output


class Discriminator_r(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.Mish(inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.Mish(inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
