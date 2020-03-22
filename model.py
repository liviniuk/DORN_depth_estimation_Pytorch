import torch
import torch.nn as nn
from torch.nn import functional as F
from resnet_dilated import resnet101dilated


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()


class SceneUnderstandingModule(nn.Module):
    def __init__(self, dataset):
        super(SceneUnderstandingModule, self).__init__()
        if dataset == 'kitti':
            dilations = [6, 12, 18]
            self.out_size = (385, 513)
        elif dataset == 'nyu':
            dilations = [4, 8, 12]
            self.out_size = (257, 353)
        
        self.encoder = FullImageEncoder(dataset)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=dilations[0], dilation=dilations[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=dilations[1], dilation=dilations[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=dilations[2], dilation=dilations[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, 160, 1) # out_channels=160=2*k for k=80; in official published models
        )
        

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print('Scene Understanding Module concat:', x.size())
        x = self.concat_process(x)
        # print('Scene Understanding Module processed:', x.size())
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=True)
        return x


class FullImageEncoder(nn.Module):
    def __init__(self, dataset):
        super(FullImageEncoder, self).__init__()
        if dataset == 'kitti':
            k = 16
            self.h, self.w = 49, 65
            self.h_, self.w_ = 4, 5
        elif dataset == 'nyu':
            k = 8
            self.h, self.w = 33, 45
            self.h_, self.w_ = 5, 6
            
        self.global_pooling = nn.AvgPool2d(k, stride=k, ceil_mode=True) # It seems, Caffe uses ceil_mode by default.
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(2048 * self.h_ * self.w_, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 conv

    def forward(self, x):
        # print('Full Image Encoder Input:', x.size())
        x = self.global_pooling(x)
        x = self.dropout(x)
        # print('Full Image Encoder Pool:', x.size())
        x = x.view(-1, 2048 * self.h_ * self.w_)
        # print('Full Image Encoder View1:', x.size())
        x = self.global_fc(x)
        x = self.relu(x)
        # print('Full Image Encoder FC:', x.size())
        x = x.view(-1, 512, 1, 1)
        # print('Full Image Encoder View2:', x.size())
        x = self.conv1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=True) # the "COPY" upsampling
        # print('Full Image Encoder Upsample:', x.size())
        return x


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N x 2K x H x W; N - batch_size, 2K - channels, K - number of discrete sub-intervals
        :return:  labels - ordinal labels (corresponding to discrete depth values) of size N x 1 x H x W
                  softmax - predicted softmax probabilities P (as in the paper) of size N x K x H x W
        """
        N, K, H, W = x.size()
        K = K // 2 # number of discrete sub-intervals
        
        odd = x[:, ::2, :, :].clone()
        even = x[:, 1::2, :, :].clone()

        odd = odd.view(N, 1, K * H * W)
        even = even.view(N, 1, K * H * W)

        paired_channels = torch.cat((odd, even), dim=1)
        paired_channels = paired_channels.clamp(min=1e-8, max=1e8)  # prevent nans

        softmax = nn.functional.softmax(paired_channels, dim=1)

        softmax = softmax[:, 1, :]
        softmax = softmax.view(-1, K, H, W)
        labels = torch.sum((softmax > 0.5), dim=1).view(-1, 1, H, W)
        return labels, softmax


class DORN(nn.Module):
    def __init__(self, dataset, pretrained=False):
        if not (dataset == 'kitti' or dataset == 'nyu'):
            raise NotImplementedError('Supported datasets: kitti | nuy (got %s)' % dataset)
            
        super(DORN, self).__init__()
        self.pretrained = pretrained
        
        self.dense_feature_extractor = resnet101dilated(pretrained=pretrained)
        self.scene_understanding_modulule = SceneUnderstandingModule(dataset=dataset)
        self.ordinal_regression = OrdinalRegressionLayer()
        
        weights_init(self.scene_understanding_modulule)
        weights_init(self.ordinal_regression)

    def forward(self, x):
        # Input image size KITTI: (385, 513), NYU: (257, 353)
        x = self.dense_feature_extractor(x) # Output KITTI: [batch, 2048, 49, 65], NYU: [batch, 2048, 33, 45].
        x = self.scene_understanding_modulule(x) # Output shape same as input shape except 2K channels.
        labels, softmax = self.ordinal_regression(x)
        return labels, softmax
    
    def train(self, mode=True):
        """
            Override train() to keep BN and first two conv layers frozend.
        """
        super().train(mode)
        
        if self.pretrained:
            # Freeze BatchNorm layers
            for module in self.modules():
                if isinstance(module, nn.modules.BatchNorm2d):
                    module.eval()

            # Freeze first two conv layers
            self.dense_feature_extractor.conv1.eval()
            self.dense_feature_extractor.conv2.eval()
        
        return self
        
    def get_1x_lr_params(self):
        for k in self.dense_feature_extractor.parameters():
            if k.requires_grad:
                yield k

    def get_10x_lr_params(self):
        for module in [self.scene_understanding_modulule, self.ordinal_regression]:
            for k in module.parameters():
                if k.requires_grad:
                    yield k