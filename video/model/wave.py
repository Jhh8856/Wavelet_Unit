''' Towards An End-to-End Framework for Video Inpainting
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.flow_comp import SPyNet
from model.modules.feat_prop import BidirectionalPropagation, SecondOrderDeformableAlignment
from model.modules.tfocal_transformer_hq import TemporalFocalTransformerBlock, SoftSplit, SoftComp
from model.modules.spectral_norm import spectral_norm as _spectral_norm

from pytorch_wavelets import DWTForward, DWTInverse

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class Wavelet(nn.Module):
    def __init__(self, feature, r1, r2, r1_2, r2_2, r3, r3_2, r4, r4_2, r5, r5_2, r6, r7):
        super().__init__()

        def batch_norm(a):
            v_h, m_h = torch.var_mean(a)
            h_batch_norm = torch.nn.functional.batch_norm(a, m_h, v_h)
            return h_batch_norm

        def Weight_Standardization(kernel, epsilon=1e-5):
            kernel_mean = torch.mean(kernel)
            kernel_std = torch.std(torch.sub(torch.kernel, kernel_mean))
            kernel_f = torch.div(kernel, (torch.add(kernel_std, epsilon)))
            return kernel_f

        self.w = DWTForward(J=3, mode='zero', wave='db3')
        self.w_i = DWTInverse(mode='zero', wave='db3')

        r1 = Weight_Standardization(r1)
        r2 = Weight_Standardization(r2)
        r1_2 = Weight_Standardization(r1_2)
        r2_2 = Weight_Standardization(r2_2)
        r3 = Weight_Standardization(r3)
        r3_2 = Weight_Standardization(r3_2)
        r4 = Weight_Standardization(r4)
        r4_2 = Weight_Standardization(r4_2)
        r5 = Weight_Standardization(r5)
        r5_2 = Weight_Standardization(r5_2)
        r6 = Weight_Standardization(r6)
        r7 = Weight_Standardization(r7)

        _, self.y_r1, self.y_r2, self.y_i = torch.split(feature, 4, dim=-1)
        self.y_r = batch_norm(torch.add(y_r1, y_r2))

        # r = tf.random.normal([1, 1, tf.shape(y_r)[-1], tf.shape(y_r)[-1]])
        self.y_r_nor = nn.conv2d(y_r, r1)
        self.y_r_nor = torch.nn.functional.leaky_relu(y_r_nor)
        # r2 = tf.random.normal([1, 1, tf.shape(y_i)[-1], tf.shape(y_i)[-1]])
        self.y_i_nor = nn.conv2d(y_i, r2)
        self.y_i_nor = torch.nn.functional.leaky_relu(y_i_nor)
        # r1_3x3 = tf.random.normal([1, 1, tf.shape(y_r)[-1], tf.shape(y_r)[-1]])
        self.y_r3x3 = nn.conv2d(y_r_nor, r1_2)
        self.y_r3x3 = torch.nn.functional.leaky_relu(y_r3x3)
        # r2_3x3 = tf.random.normal([1, 1, tf.shape(y_i)[-1], tf.shape(y_i)[-1]])
        self.y_i3x3 = nn.conv2d(y_r_nor, r2_2)
        self.y_i3x3 = torch.nn.functional.leaky_relu(y_i3x3)
        self.freq_y_i_low, self.y_i = w(y_i_nor)
        # print(freq_y_i)
        freq_y_i_mid1, freq_y_i_mid2, freq_y_i_high = torch.split(freq_y_i, 3, dim=2)
        # freq_y_i_mid = tf.concat((freq_y_i_mid1, freq_y_i_mid2), axis=-1)
        #print(freq_y_i_low)
        #print(freq_y_i_mid1)
        #print(freq_y_i_high)

        self.y_sm1 = freq_y_i_mid1
        # r3 = tf.random.normal([1, 1, tf.shape(y_sm)[-1], tf.shape(y_sm)[-1]])
        self.y_sm1 = conv2d(y_sm1, r3)
        self.y_sm1 = conv2d(y_sm1, r3_2)
        self.y_sm1 = torch.nn.functional.leaky_relu(y_sm1_2)

        self.y_sm2 = freq_y_i_mid1
        # r3 = tf.random.normal([1, 1, tf.shape(y_sm)[-1], tf.shape(y_sm)[-1]])
        self.y_sm2 = conv2d(y_sm2, r4)
        self.y_sm2 = conv2d(y_sm2, r4_2)
        self.y_sm2 = torch.nn.functional.leaky_relu(y_sm2)

        self.y_sh = freq_y_i_high
        # r4 = tf.random.normal([1, 1, tf.shape(y_sh)[-1], tf.shape(y_sh)[-1]])
        self.y_sh = tf.nn.conv2d(y_sh, r5)
        self.y_sh = tf.nn.conv2d(y_sh, r5_2)
        self.y_sh = torch.nn.functional.leaky_relu(y_sh)

        #print(freq_y_i_low)
        #print(y_sm1)
        #print(y_sh)
        self.y_s = torch.concat((freq_y_i_low, y_sm1, y_sm2, y_sh), dim=-1)
        #print(y_s)
        # z_r, z_i = tf.split(y_s, num_or_size_splits=2, axis=-1)
        # z = tf.dtypes.complex(z_r, z_i)
        # z = tf.signal.irfft2d(z)
        # w_i = WaveTFFactory().build('db2', dim=2, inverse=True)
        self.z = w_i(y_s)
        # r5 = tf.random.normal([1, 1, tf.shape(z)[-1], tf.shape(z)[-1]])
        self.spectral = conv2d(z, r6)
        self.spectral = torch.nn.functional.leaky_relu(spectral)

        self.global_feature = torch.add(y_r3x3, spectral)
        self.global_feature = torch.nn.functional.leaky_relu(global_feature)

        self.local_feature = torch.add(y_r3x3, y_i3x3)
        self.local_feature = torch.nn.functional.leaky_relu(local_feature)

        self.final_a = torch.concat((global_feature, local_feature), dim=-1)
        self.final_conv = conv2d(final_a, r7)
        self.final = tf.nn.leaky_relu(final_conv)

    def forward(self, x):
        x = F.interpolate(x, mode='bilinear', align_corners=True)

        return self.final(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.r1 = torch.nn.Parameter(requires_grad=True)
        self.r2 = torch.nn.Parameter(requires_grad=True)
        self.r1_2 = torch.nn.Parameter(requires_grad=True)
        self.r2_2 = torch.nn.Parameter(requires_grad=True)
        self.r3 = torch.nn.Parameter(requires_grad=True)
        self.r3_2 = torch.nn.Parameter(requires_grad=True)
        self.r4 = torch.nn.Parameter(requires_grad=True)
        self.r4_2 = torch.nn.Parameter(requires_grad=True)
        self.r5 = torch.nn.Parameter(requires_grad=True)
        self.r5_2 = torch.nn.Parameter(requires_grad=True)
        self.r6 = torch.nn.Parameter(requires_grad=True)
        self.r7 = torch.nn.Parameter(requires_grad=True)

        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        out = Wavelet(out, r1, r2, r1_2, r2_2,
                      r3, r3_2, r4, r4_2, r5,
                      r5_2, r6, r7)
        return out


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        channel = 256
        hidden = 512

        # encoder
        self.encoder = Encoder()

        # decoder
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # feature propagation module
        self.feat_prop_module = BidirectionalPropagation(channel // 2)

        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        self.ss = SoftSplit(channel // 2,
                            hidden,
                            kernel_size,
                            stride,
                            padding,
                            t2t_param=t2t_params)
        self.sc = SoftComp(channel // 2, hidden, kernel_size, stride, padding)

        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] -
                           (d - 1) - 1) / stride[i] + 1)

        blocks = []
        depths = 8
        num_heads = [4] * depths
        window_size = [(5, 9)] * depths
        focal_windows = [(5, 9)] * depths
        focal_levels = [2] * depths
        pool_method = "fc"

        for i in range(depths):
            blocks.append(
                TemporalFocalTransformerBlock(dim=hidden,
                                              num_heads=num_heads[i],
                                              window_size=window_size[i],
                                              focal_level=focal_levels[i],
                                              focal_window=focal_windows[i],
                                              n_vecs=n_vecs,
                                              t2t_params=t2t_params,
                                              pool_method=pool_method))
        self.transformer = nn.Sequential(*blocks)

        if init_weights:
            self.init_weights()
            # Need to initial the weights of MSDeformAttn specifically
            for m in self.modules():
                if isinstance(m, SecondOrderDeformableAlignment):
                    m.init_offset()

        # flow completion network
        self.update_spynet = SPyNet()

    def forward_bidirect_flow(self, masked_local_frames):
        b, l_t, c, h, w = masked_local_frames.size()

        # compute forward and backward flows of masked frames
        masked_local_frames = F.interpolate(masked_local_frames.view(
            -1, c, h, w),
                                            scale_factor=1 / 4,
                                            mode='bilinear',
                                            align_corners=True,
                                            recompute_scale_factor=True)
        masked_local_frames = masked_local_frames.view(b, l_t, c, h // 4,
                                                       w // 4)
        mlf_1 = masked_local_frames[:, :-1, :, :, :].reshape(
            -1, c, h // 4, w // 4)
        mlf_2 = masked_local_frames[:, 1:, :, :, :].reshape(
            -1, c, h // 4, w // 4)
        pred_flows_forward = self.update_spynet(mlf_1, mlf_2)
        pred_flows_backward = self.update_spynet(mlf_2, mlf_1)

        pred_flows_forward = pred_flows_forward.view(b, l_t - 1, 2, h // 4,
                                                     w // 4)
        pred_flows_backward = pred_flows_backward.view(b, l_t - 1, 2, h // 4,
                                                       w // 4)

        return pred_flows_forward, pred_flows_backward

    def forward(self, masked_frames, num_local_frames):
        l_t = num_local_frames
        b, t, ori_c, ori_h, ori_w = masked_frames.size()

        # normalization before feeding into the flow completion module
        masked_local_frames = (masked_frames[:, :l_t, ...] + 1) / 2
        pred_flows = self.forward_bidirect_flow(masked_local_frames)

        # extracting features and performing the feature propagation on local features
        enc_feat = self.encoder(masked_frames.view(b * t, ori_c, ori_h, ori_w))
        _, c, h, w = enc_feat.size()
        fold_output_size = (h, w)
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        local_feat = self.feat_prop_module(local_feat, pred_flows[0],
                                           pred_flows[1])
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        # content hallucination through stacking multiple temporal focal transformer blocks
        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_output_size)
        trans_feat = self.transformer([trans_feat, fold_output_size])
        trans_feat = self.sc(trans_feat[0], t, fold_output_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)
        enc_feat = enc_feat + trans_feat

        # decode frames from features
        output = self.decoder(enc_feat.view(b * t, c, h, w))
        output = torch.tanh(output)
        return output, pred_flows


# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
