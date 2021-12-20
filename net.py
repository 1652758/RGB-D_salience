from model.module import *
from model.unet import DoubleConv, OutConv
import numpy as np


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.encoder_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, input):
        return self.encoder_conv(input)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, input):
        return self.decoder_conv(input)


class AT_GEN_Block(nn.Module):
    def __init__(self, upsampling_rate=1, pow_coeff=2):
        super(AT_GEN_Block, self).__init__()
        self.pow_coeff = pow_coeff
        self.up = nn.Upsample(scale_factor=upsampling_rate, mode='bilinear', align_corners=True)

    def forward(self, input):
        x = self.up(input)
        x_sum = torch.sum(torch.pow(torch.abs(x), self.pow_coeff), dim=1, keepdim=True)
        def spatial_soft_max(x):
            org_shape = x.shape
            x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # batch, C, W*H
            x = F.softmax(x, -1)                                          # batch, C, W*H
            x = x.view(org_shape[0], org_shape[1], org_shape[2], org_shape[3])
            return x
        return spatial_soft_max(x_sum)


class FusionBlock(nn.Module):
    def __init__(self):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.ac = nn.Sigmoid()

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, input_m1, input_m2):
        input = torch.cat([input_m1, input_m2], dim=1)
        shuffled_x = self.channel_shuffle(input, 2)
        x = shuffled_x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(1)
        return torch.add(torch.mul(self.ac(x), shuffled_x), shuffled_x)





class SCFFBlock(nn.Module):
    def __init__(self, in_c):
        super(SCFFBlock, self).__init__()
        self.conv_out_dim = 64
        self.conv = Conv2D(in_c, self.conv_out_dim, 1, bias=False, bn=True, ac=True)
        self.conv_out = Conv2D(in_c*2, in_c*2, 1, bias=False, bn=True, ac=True)

    def spatial_soft_max(self, x):
        org_shape = x.shape
        x = x.view(x.shape[0], -1)  # batch, C, W*H
        x = F.softmax(x, -1)  # batch, C, W*H
        x = x.view(org_shape[0], org_shape[1], org_shape[2])
        return x

    def scff_opertaion(self, x1, x2):
        batchsize, num_channels, height, width = x1.data.size()
        f_1 = self.conv(x1)
        f_2 = self.conv(x2)
        f_m_1 = f_1.view(batchsize, self.conv_out_dim, -1)
        f_m_2 = f_2.view(batchsize, self.conv_out_dim, -1)
        co_map = torch.bmm(torch.transpose(f_m_1, 1, 2), f_m_2)
        # co_map = self.spatial_soft_max(co_map)
        mul_x1 = x1.view(batchsize, num_channels, -1)
        mul_x2 = x2.view(batchsize, num_channels, -1)
        f_map = torch.cat([torch.bmm(mul_x1, co_map), torch.bmm(mul_x2, torch.transpose(co_map, 1, 2))], dim=1)
        return self.conv_out(f_map.view(batchsize, num_channels*2, height, width))

    def forward(self, input_m1, input_m2):
        return self.scff_opertaion(input_m1, input_m2)





class SADNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, channel_unit=16, shuffled_fusion=True):
        super(SADNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.inc_m1 = DoubleConv(n_channels, channel_unit)
        self.inc_m2 = DoubleConv(n_channels, channel_unit)

        self.down_m1_1 = EncoderBlock(channel_unit, channel_unit * 2)
        self.down_m1_2 = EncoderBlock(channel_unit * 2, channel_unit * 4)
        self.down_m1_3 = EncoderBlock(channel_unit * 4, channel_unit * 8)
        self.down_m1_4 = EncoderBlock(channel_unit * 8, channel_unit * 8)

        self.down_m2_1 = EncoderBlock(channel_unit, channel_unit * 2)
        self.down_m2_2 = EncoderBlock(channel_unit * 2, channel_unit * 4)
        self.down_m2_3 = EncoderBlock(channel_unit * 4, channel_unit * 8)
        self.down_m2_4 = EncoderBlock(channel_unit * 8, channel_unit * 8)

        self.at_gen_m1_1 = AT_GEN_Block(upsampling_rate=1)
        self.at_gen_m1_2 = AT_GEN_Block(upsampling_rate=2)
        self.at_gen_m1_3 = AT_GEN_Block(upsampling_rate=4)
        self.at_gen_m1_4 = AT_GEN_Block(upsampling_rate=8)

        self.at_gen_m2_1 = AT_GEN_Block(upsampling_rate=1)
        self.at_gen_m2_2 = AT_GEN_Block(upsampling_rate=2)
        self.at_gen_m2_3 = AT_GEN_Block(upsampling_rate=4)
        self.at_gen_m2_4 = AT_GEN_Block(upsampling_rate=8)



        self.fusion = FusionBlock() if shuffled_fusion else None

        self.up = nn.Sequential(
            DecoderBlock(channel_unit * 16, channel_unit * 8),
            DecoderBlock(channel_unit * 8, channel_unit * 4),
            DecoderBlock(channel_unit * 4, channel_unit * 2),
            DecoderBlock(channel_unit * 2, channel_unit),
            OutConv(channel_unit, n_classes)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_m1, input_m2):
        x_m1_1 = self.inc_m1(input_m1)
        x_m1_2 = self.down_m1_1(x_m1_1)
        x_m1_3 = self.down_m1_2(x_m1_2)
        x_m1_4 = self.down_m1_3(x_m1_3)
        x_m1_5 = self.down_m1_4(x_m1_4)

        at_m1_1 = self.at_gen_m1_1(x_m1_2)
        at_m1_2 = self.at_gen_m1_2(x_m1_3)
        at_m1_3 = self.at_gen_m1_3(x_m1_4)
        at_m1_4 = self.at_gen_m1_4(x_m1_5)

        x_m2_1 = self.inc_m2(input_m2)
        x_m2_2 = self.down_m2_1(x_m2_1)
        x_m2_3 = self.down_m2_2(x_m2_2)
        x_m2_4 = self.down_m2_3(x_m2_3)
        x_m2_5 = self.down_m2_4(x_m2_4)

        at_m2_1 = self.at_gen_m2_1(x_m2_2)
        at_m2_2 = self.at_gen_m2_2(x_m2_3)
        at_m2_3 = self.at_gen_m2_3(x_m2_4)
        at_m2_4 = self.at_gen_m2_4(x_m2_5)
        if self.fusion is not None:
            x = self.fusion(x_m1_5, x_m2_5)
        else:
            x = torch.cat([x_m1_5, x_m2_5], dim=1)
        x = self.up(x)

        return x, [at_m1_1, at_m1_2, at_m1_3, at_m1_4], [at_m2_1, at_m2_2, at_m2_3, at_m2_4]




class SADNetWithSCFF(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, channel_unit=32, scff_fusion=True):
        super(SADNetWithSCFF, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.inc_m1 = DoubleConv(n_channels, channel_unit)
        self.inc_m2 = DoubleConv(n_channels, channel_unit)

        self.down_m1_1 = EncoderBlock(channel_unit, channel_unit * 2)
        self.down_m1_2 = EncoderBlock(channel_unit * 2, channel_unit * 4)
        self.down_m1_3 = EncoderBlock(channel_unit * 4, channel_unit * 8)
        self.down_m1_4 = EncoderBlock(channel_unit * 8, channel_unit * 8)

        self.down_m2_1 = EncoderBlock(channel_unit, channel_unit * 2)
        self.down_m2_2 = EncoderBlock(channel_unit * 2, channel_unit * 4)
        self.down_m2_3 = EncoderBlock(channel_unit * 4, channel_unit * 8)
        self.down_m2_4 = EncoderBlock(channel_unit * 8, channel_unit * 8)

        self.at_gen_m1_1 = AT_GEN_Block(upsampling_rate=1)
        self.at_gen_m1_2 = AT_GEN_Block(upsampling_rate=2)
        self.at_gen_m1_3 = AT_GEN_Block(upsampling_rate=4)
        self.at_gen_m1_4 = AT_GEN_Block(upsampling_rate=8)

        self.at_gen_m2_1 = AT_GEN_Block(upsampling_rate=1)
        self.at_gen_m2_2 = AT_GEN_Block(upsampling_rate=2)
        self.at_gen_m2_3 = AT_GEN_Block(upsampling_rate=4)
        self.at_gen_m2_4 = AT_GEN_Block(upsampling_rate=8)

        self.fusion = SCFFBlock(channel_unit * 8) if scff_fusion else None

        self.up = nn.Sequential(
            DecoderBlock(channel_unit * 16, channel_unit * 8),
            DecoderBlock(channel_unit * 8, channel_unit * 4),
            DecoderBlock(channel_unit * 4, channel_unit * 2),
            DecoderBlock(channel_unit * 2, channel_unit),
            OutConv(channel_unit, n_classes)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_m1, input_m2):
        x_m1_1 = self.inc_m1(input_m1)
        x_m1_2 = self.down_m1_1(x_m1_1)
        x_m1_3 = self.down_m1_2(x_m1_2)
        x_m1_4 = self.down_m1_3(x_m1_3)
        x_m1_5 = self.down_m1_4(x_m1_4)

        at_m1_1 = self.at_gen_m1_1(x_m1_2)
        at_m1_2 = self.at_gen_m1_2(x_m1_3)
        at_m1_3 = self.at_gen_m1_3(x_m1_4)
        at_m1_4 = self.at_gen_m1_4(x_m1_5)

        x_m2_1 = self.inc_m2(input_m2)
        x_m2_2 = self.down_m2_1(x_m2_1)
        x_m2_3 = self.down_m2_2(x_m2_2)
        x_m2_4 = self.down_m2_3(x_m2_3)
        x_m2_5 = self.down_m2_4(x_m2_4)

        at_m2_1 = self.at_gen_m2_1(x_m2_2)
        at_m2_2 = self.at_gen_m2_2(x_m2_3)
        at_m2_3 = self.at_gen_m2_3(x_m2_4)
        at_m2_4 = self.at_gen_m2_4(x_m2_5)
        if self.fusion is not None:
            x = self.fusion(x_m1_5, x_m2_5)
        else:
            x = torch.cat([x_m1_5, x_m2_5], dim=1)
        x = self.up(x)

        return x, [at_m1_1, at_m1_2, at_m1_3, at_m1_4], [at_m2_1, at_m2_2, at_m2_3, at_m2_4]






class SADNetWithEarlyFusion(nn.Module):
    def __init__(self, n_channels=2, n_classes=1, bilinear=True, channel_unit=64):
        super(SADNetWithEarlyFusion, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, channel_unit)

        self.down = nn.Sequential(
            EncoderBlock(channel_unit, channel_unit * 2),
            EncoderBlock(channel_unit * 2, channel_unit * 4),
            EncoderBlock(channel_unit * 4, channel_unit * 8),
            EncoderBlock(channel_unit * 8, channel_unit * 8)
        )

        self.up = nn.Sequential(
            DecoderBlock(channel_unit * 8, channel_unit * 8),
            DecoderBlock(channel_unit * 8, channel_unit * 4),
            DecoderBlock(channel_unit * 4, channel_unit * 2),
            DecoderBlock(channel_unit * 2, channel_unit),
            OutConv(channel_unit, n_classes)
        )

    def forward(self, input_m1, input_m2):
        x = torch.cat([input_m1, input_m2], dim=1)
        x = self.inc(x)
        x = self.down(x)
        logits = self.up(x)
        return logits


class SADNetWithLateFusion(nn.Module):
    def __init__(self, n_channels=2, n_classes=1, bilinear=True, channel_unit=32):
        super(SADNetWithLateFusion, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_m1 = DoubleConv(n_channels, channel_unit)
        self.inc_m2 = DoubleConv(n_channels, channel_unit)

        self.down_m1_1 = EncoderBlock(channel_unit, channel_unit * 2)
        self.down_m1_2 = EncoderBlock(channel_unit * 2, channel_unit * 4)
        self.down_m1_3 = EncoderBlock(channel_unit * 4, channel_unit * 8)
        self.down_m1_4 = EncoderBlock(channel_unit * 8, channel_unit * 8)

        self.down_m2_1 = EncoderBlock(channel_unit, channel_unit * 2)
        self.down_m2_2 = EncoderBlock(channel_unit * 2, channel_unit * 4)
        self.down_m2_3 = EncoderBlock(channel_unit * 4, channel_unit * 8)
        self.down_m2_4 = EncoderBlock(channel_unit * 8, channel_unit * 8)

        self.at_gen_m1_1 = AT_GEN_Block(upsampling_rate=1)
        self.at_gen_m1_2 = AT_GEN_Block(upsampling_rate=2)
        self.at_gen_m1_3 = AT_GEN_Block(upsampling_rate=4)
        self.at_gen_m1_4 = AT_GEN_Block(upsampling_rate=8)

        self.at_gen_m2_1 = AT_GEN_Block(upsampling_rate=1)
        self.at_gen_m2_2 = AT_GEN_Block(upsampling_rate=2)
        self.at_gen_m2_3 = AT_GEN_Block(upsampling_rate=4)
        self.at_gen_m2_4 = AT_GEN_Block(upsampling_rate=8)

        self.up_m1 = nn.Sequential(
            DecoderBlock(channel_unit * 8, channel_unit * 8),
            DecoderBlock(channel_unit * 8, channel_unit * 4),
            DecoderBlock(channel_unit * 4, channel_unit * 2),
            DecoderBlock(channel_unit * 2, channel_unit)
        )

        self.up_m2 = nn.Sequential(
            DecoderBlock(channel_unit * 8, channel_unit * 8),
            DecoderBlock(channel_unit * 8, channel_unit * 4),
            DecoderBlock(channel_unit * 4, channel_unit * 2),
            DecoderBlock(channel_unit * 2, channel_unit)
        )

        self.fusion = FusionBlock()

        self.out_conv = OutConv(channel_unit * 2, n_classes)

    def forward(self, input_m1, input_m2):
        x_m1_1 = self.inc_m1(input_m1)
        x_m1_2 = self.down_m1_1(x_m1_1)
        x_m1_3 = self.down_m1_2(x_m1_2)
        x_m1_4 = self.down_m1_3(x_m1_3)
        x_m1_5 = self.down_m1_4(x_m1_4)

        at_m1_1 = self.at_gen_m1_1(x_m1_2)
        at_m1_2 = self.at_gen_m1_2(x_m1_3)
        at_m1_3 = self.at_gen_m1_3(x_m1_4)
        at_m1_4 = self.at_gen_m1_4(x_m1_5)

        x_m2_1 = self.inc_m2(input_m2)
        x_m2_2 = self.down_m2_1(x_m2_1)
        x_m2_3 = self.down_m2_2(x_m2_2)
        x_m2_4 = self.down_m2_3(x_m2_3)
        x_m2_5 = self.down_m2_4(x_m2_4)

        at_m2_1 = self.at_gen_m2_1(x_m2_2)
        at_m2_2 = self.at_gen_m2_2(x_m2_3)
        at_m2_3 = self.at_gen_m2_3(x_m2_4)
        at_m2_4 = self.at_gen_m2_4(x_m2_5)

        xm1 = self.up_m1(x_m1_5)
        xm2 = self.up_m2(x_m2_5)

        x = self.fusion(xm1, xm2)

        logits = self.out_conv(x)

        return logits, [at_m1_1, at_m1_2, at_m1_3, at_m1_4], [at_m2_1, at_m2_2, at_m2_3, at_m2_4]