import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Spatial import Spatial
from model.Flow import Flow
from model.Depth import Depth
from torchvision.transforms import Resize

# Pixel-level Selective Fusion Strategy
class PSF(nn.Module):
    def __init__(self):
        super(PSF, self).__init__()
        self.decoder5 = decoder_stage(64, 128, 64)  #
        self.decoder4 = decoder_stage(128, 128, 64)  #
        self.decoder3 = decoder_stage(128, 128, 64)  #
        self.decoder2 = decoder_stage(128, 128, 64)  #
        self.decoder1 = decoder_stage(128, 128, 64)  #

        self.out5 = out_block(64, sig=True)
        self.out4 = out_block(64, sig=True)
        self.out3 = out_block(64, sig=True)
        self.out2 = out_block(64, sig=True)
        self.out1 = out_block(64, sig=True)

    def forward(self, fd):
        out1, out2, out3, out4, out5 = fd[0],fd[1],fd[2],fd[3],fd[4]
        feature5 = self.decoder5(out5)
        feature4 = self.decoder4(torch.cat([feature5, out4], 1))
        B, C, H, W = out3.size()
        feature3 = self.decoder3(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), out3), 1))
        B, C, H, W = out2.size()
        feature2 = self.decoder2(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), out2), 1))
        B, C, H, W = out1.size()
        feature1 = self.decoder1(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), out1), 1))

        decoder_out1 = self.out1(feature1, H * 4, W * 4)

        return decoder_out1

# Multi-dimension Selective Attention Module
class MSAM(nn.Module):
    def __init__(self,inplanes):
        super(MSAM,self).__init__()
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.conv_h1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.conv_h2 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)

        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_w = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.conv_w1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)

        self.pool_glo = nn.AdaptiveAvgPool2d(1)
        self.conv_glo = nn.Conv2d(inplanes * 2, inplanes * 2, kernel_size=1, stride=1, padding=0)
        self.conv_glo1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.conv_glo2 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)

        self.conv_c = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)

    def forward(self,rgb,fd):
        rfd = rgb * fd
        en_rgb = self.sa1(rfd) * rgb + rgb
        en_fd = self.sa2(rfd) * fd + fd

        B, C, H, W = en_rgb.size()
        h_rgb = self.pool_h(en_rgb)
        h_fd = self.pool_h(en_fd)
        h = torch.cat([h_rgb, h_fd], dim=2)
        h = self.conv_h(h)
        h1, h2 = torch.split(h, [H, H], dim=2)
        h_out = self.sigmoid(self.conv_h1(h1)) * en_rgb + self.sigmoid(self.conv_h2(h2)) * en_fd

        w_rgb = self.pool_w(en_rgb)
        w_fd = self.pool_w(en_fd)
        w = torch.cat([w_rgb, w_fd], dim=3)
        w = self.conv_w(w)
        w1, w2 = torch.split(w, [W, W], dim=3)
        w_out = self.sigmoid(self.conv_w1(w1)) * en_rgb + self.sigmoid(self.conv_w2(w2)) * en_fd

        glo_rgb = self.pool_glo(en_rgb)
        glo_fd = self.pool_glo(en_fd)
        glo = torch.cat([glo_rgb,glo_fd],dim=1)
        glo = self.conv_glo(glo)
        glo1, glo2 = torch.split(glo, [C, C], dim=1)
        glo_out = self.sigmoid(self.conv_glo1(glo1)) * en_rgb + self.sigmoid(self.conv_glo2(glo2)) * en_fd

        c_rgb = torch.mean(en_rgb, dim=1, keepdim=True)
        c_fd =  torch.mean(en_fd, dim=1, keepdim=True)
        c = torch.cat([c_rgb,c_fd],dim=1)
        c = self.conv_c(c)
        c = F.softmax(c, dim=1)
        c1, c2 = torch.split(c, [1, 1], dim=1)
        c_out = c1 * en_rgb + c2 * en_fd

        out = h_out + w_out + glo_out + c_out
        out = self.conv_out(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class REM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(REM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class out_block(nn.Module):
    def __init__(self, infilter,sig=False):
        super(out_block, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(infilter, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        if sig==False:
            self.conv2 = nn.Conv2d(64, 1, 1)
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(64, 1, 1),nn.Sigmoid())


    def forward(self, x, H, W):
        x = F.interpolate(self.conv1(x), (H, W), mode='bilinear', align_corners=True)
        return self.conv2(x)


class decoder_stage(nn.Module):
    def __init__(self, infilter, midfilter, outfilter):
        super(decoder_stage, self).__init__()
        self.layer = nn.Sequential(
            *[nn.Conv2d(infilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              nn.Conv2d(midfilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              nn.Conv2d(midfilter, outfilter, 3, padding=1, bias=False), nn.BatchNorm2d(outfilter),
              nn.ReLU(inplace=True)])

    def forward(self, x):
        return self.layer(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class Model(nn.Module):
    def __init__(self, inchannels, mode, spatial_ckpt=None, flow_ckpt=None, depth_ckpt=None):
        super(Model, self).__init__()
        self.spatial_net = Spatial(inchannels, mode)
        self.flow_net = Flow(inchannels, mode)
        self.depth_net = Depth(inchannels, mode)

        self.fdconv1 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fdconv2 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fdconv3 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fdconv4 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fdconv5 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.psf = PSF()

        self.msam1 = MSAM(64)
        self.msam2 = MSAM(64)
        self.msam3 = MSAM(64)
        self.msam4 = MSAM(64)
        self.msam5 = MSAM(64)

        self.rem1 = REM(64, 64)
        self.rem2 = REM(64, 64)
        self.rem3 = REM(64, 64)
        self.rem4 = REM(64, 64)
        self.rem5 = REM(64, 64)

        self.inplanes = 32
        self.deconv1 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.inplanes = 16
        self.deconv2 = self._make_transpose(TransBasicBlock, 16, 3, stride=2)
        self.agant1 = self._make_agant_layer(64, 32)
        self.agant2 = self._make_agant_layer(32, 16)
        self.outconv2 = nn.Conv2d(16 * 1, 1, kernel_size=1, stride=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if spatial_ckpt is not None:
            self.spatial_net.load_state_dict(torch.load(spatial_ckpt, map_location='cpu'))
            print("Successfully load spatial:{}".format(spatial_ckpt))
        if flow_ckpt is not None:
            self.flow_net.load_state_dict(torch.load(flow_ckpt, map_location='cpu'))
            print("Successfully load flow:{}".format(flow_ckpt))
        if depth_ckpt is not None:
            self.depth_net.load_state_dict(torch.load(depth_ckpt, map_location='cpu'))
            print("Successfully load depth:{}".format(depth_ckpt))



    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, image, flow, depth):
        out4, out1, img_conv1_feat, out2, out3, out5_aspp, course_img = self.spatial_net.rgb_bkbone(image)

        flow_out4, flow_out1, flow_conv1_feat, flow_out2, flow_out3, flow_out5_aspp, course_flo = self.flow_net.flow_bkbone(
            flow)

        depth_out4, depth_out1, depth_conv1_feat, depth_out2, depth_out3, depth_out5_aspp, course_dep = self.depth_net.depth_bkbone(
            depth)

        #Pixel-level Selective Fusion Strategy
        flow_out1_squ, flow_out2_squ, flow_out3_squ, flow_out4_squ, flow_out5_squ = self.flow_net.squeeze1_flow(
            flow_out1), \
            self.flow_net.squeeze2_flow(flow_out2), \
            self.flow_net.squeeze3_flow(flow_out3), \
            self.flow_net.squeeze4_flow(flow_out4), \
            self.flow_net.squeeze5_flow(flow_out5_aspp)

        flow_feature5 = self.flow_net.decoder5_flow(flow_out5_squ)
        flow_feature4 = self.flow_net.decoder4_flow(torch.cat([flow_feature5, flow_out4_squ], 1))
        B, C, H, W = flow_out3.size()
        flow_feature3 = self.flow_net.decoder3_flow(
            torch.cat((F.interpolate(flow_feature4, (H, W), mode='bilinear', align_corners=True), flow_out3_squ), 1))
        B, C, H, W = flow_out2.size()
        flow_feature2 = self.flow_net.decoder2_flow(
            torch.cat((F.interpolate(flow_feature3, (H, W), mode='bilinear', align_corners=True), flow_out2_squ), 1))
        B, C, H, W = flow_out1.size()
        flow_feature1 = self.flow_net.decoder1_flow(
            torch.cat((F.interpolate(flow_feature2, (H, W), mode='bilinear', align_corners=True), flow_out1_squ), 1))

        decoder_out1_flow = self.flow_net.out1_flow(flow_feature1, H * 4, W * 4)
        decoder_out1_flow = nn.Sigmoid()(decoder_out1_flow)

        depth_out1_squ, depth_out2_squ, depth_out3_squ, depth_out4_squ, depth_out5_squ = self.depth_net.squeeze1_depth(
            depth_out1), \
            self.depth_net.squeeze2_depth(depth_out2), \
            self.depth_net.squeeze3_depth(depth_out3), \
            self.depth_net.squeeze4_depth(depth_out4), \
            self.depth_net.squeeze5_depth(depth_out5_aspp)

        depth_feature5 = self.depth_net.decoder5_depth(depth_out5_squ)
        depth_feature4 = self.depth_net.decoder4_depth(torch.cat([depth_feature5, depth_out4_squ], 1))
        B, C, H, W = depth_out3.size()
        depth_feature3 = self.depth_net.decoder3_depth(
            torch.cat((F.interpolate(depth_feature4, (H, W), mode='bilinear', align_corners=True), depth_out3_squ), 1))
        B, C, H, W = depth_out2.size()
        depth_feature2 = self.depth_net.decoder2_depth(
            torch.cat((F.interpolate(depth_feature3, (H, W), mode='bilinear', align_corners=True), depth_out2_squ), 1))
        B, C, H, W = depth_out1.size()
        depth_feature1 = self.depth_net.decoder1_depth(
            torch.cat((F.interpolate(depth_feature2, (H, W), mode='bilinear', align_corners=True), depth_out1_squ), 1))

        decoder_out1_depth = self.depth_net.out1_depth(depth_feature1, H * 4, W * 4)
        decoder_out1_depth = nn.Sigmoid()(decoder_out1_depth)

        fd1 = self.fdconv1(torch.cat((flow_out1_squ, depth_out1_squ), dim=1))
        fd2 = self.fdconv2(torch.cat((flow_out2_squ, depth_out2_squ), dim=1))
        fd3 = self.fdconv3(torch.cat((flow_out3_squ, depth_out3_squ), dim=1))
        fd4 = self.fdconv4(torch.cat((flow_out4_squ, depth_out4_squ), dim=1))
        fd5 = self.fdconv5(torch.cat((flow_out5_squ, depth_out5_squ), dim=1))
        sw_f = self.psf([fd1,fd2,fd3,fd4,fd5])

        B, C, H, W = flow_out1.size()
        swf1 = F.interpolate(sw_f, (H, W), mode='bilinear', align_corners=True)

        B, C, H, W = flow_out2.size()
        swf2 = F.interpolate(sw_f, (H, W), mode='bilinear', align_corners=True)

        B, C, H, W = flow_out3.size()
        swf3 = F.interpolate(sw_f, (H, W), mode='bilinear', align_corners=True)

        B, C, H, W = flow_out4.size()
        swf4 = F.interpolate(sw_f, (H, W), mode='bilinear', align_corners=True)

        B, C, H, W = flow_out5_aspp.size()
        swf5 = F.interpolate(sw_f, (H, W), mode='bilinear', align_corners=True)

        m1 = swf1 * flow_out1_squ + (1-swf1) * depth_out1_squ
        m2 = swf2 * flow_out2_squ + (1-swf2) * depth_out2_squ
        m3 = swf3 * flow_out3_squ + (1-swf3) * depth_out3_squ
        m4 = swf4 * flow_out4_squ + (1-swf4) * depth_out4_squ
        m5 = swf5 * flow_out5_squ + (1-swf5) * depth_out5_squ

        out1, out2, out3, out4, out5 = self.spatial_net.squeeze1(out1), \
            self.spatial_net.squeeze2(out2), \
            self.spatial_net.squeeze3(out3), \
            self.spatial_net.squeeze4(out4), \
            self.spatial_net.squeeze5(out5_aspp)

        # Multi-dimension Selective Attention Module
        fusion1 = self.msam1(out1, m1)
        fusion2 = self.msam2(out2, m2)
        fusion3 = self.msam3(out3, m3)
        fusion4 = self.msam4(out4, m4)
        fusion5 = self.msam5(out5, m5)

        feature5 = self.spatial_net.decoder5(fusion5)
        feature5 = self.rem5(feature5)

        feature4 = self.spatial_net.decoder4(torch.cat([feature5, fusion4], 1))
        feature4 = self.rem4(feature4)

        B, C, H, W = fusion3.size()
        feature3 = self.spatial_net.decoder3(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), fusion3), 1))
        feature3 = self.rem3(feature3)

        B, C, H, W = fusion2.size()
        feature2 = self.spatial_net.decoder2(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), fusion2), 1))
        feature2 = self.rem2(feature2)

        B, C, H, W = fusion1.size()
        feature1 = self.spatial_net.decoder1(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), fusion1), 1))
        feature1 = self.rem1(feature1)

        decoder_out5 = self.spatial_net.out5(feature5, H * 4, W * 4)
        decoder_out4 = self.spatial_net.out4(feature4, H * 4, W * 4)
        decoder_out3 = self.spatial_net.out3(feature3, H * 4, W * 4)
        decoder_out2 = self.spatial_net.out2(feature2, H * 4, W * 4)

        y = self.agant1(feature1)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        decoder_out1 = self.outconv2(y)

        return [decoder_out1,decoder_out2,decoder_out3,decoder_out4,decoder_out5],decoder_out1_flow,decoder_out1_depth,sw_f

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

if __name__ == '__main__':
    model = Model(3, 'train')
    image = flow = depth = torch.ones([2, 3, 448, 448])
    decoder_out = model(image, flow, depth)
    print(model)