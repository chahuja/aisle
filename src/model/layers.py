import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pdb
import copy

import torch
import torch.nn as nn
from transformers import BertModel
import logging
logging.getLogger('transformers').setLevel(logging.CRITICAL)

def num_powers_of_two(x):
  num_powers = 0
  while x>1:
    if x % 2 == 0:
      x /= 2
      num_powers += 1
    else:
      break
  return num_powers

def next_multiple_power_of_two(x, power=5):
  curr_power = num_powers_of_two(x)
  if curr_power < power:
    x = x * (2**(power-curr_power))
  return x

class ConvNormRelu(nn.Module):
  def __init__(self, in_channels, out_channels,
               type='1d', leaky=False,
               downsample=False, kernel_size=None, stride=None,
               padding=None, p=0, groups=1):
    super(ConvNormRelu, self).__init__()
    if kernel_size is None and stride is None:
      if not downsample:
        kernel_size = 3
        stride = 1
      else:
        kernel_size = 4
        stride = 2

    if padding is None:
      if isinstance(kernel_size, int) and isinstance(stride, tuple):
        padding = tuple(int((kernel_size - st)/2) for st in stride)
      elif isinstance(kernel_size, tuple) and isinstance(stride, int):
        padding = tuple(int((ks - stride)/2) for ks in kernel_size)
      elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
        assert len(kernel_size) == len(stride), 'dims in kernel_size are {} and stride are {}. They must be the same'.format(len(kernel_size), len(stride))
        padding = tuple(int((ks - st)/2) for ks, st in zip(kernel_size, kernel_size))
      else:
        padding = int((kernel_size - stride)/2)


    in_channels = in_channels*groups
    out_channels = out_channels*groups
    if type == '1d':
      self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm1d(out_channels)
      self.dropout = nn.Dropout(p=p)
    elif type == '2d':
      self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm2d(out_channels)
      self.dropout = nn.Dropout2d(p=p)
    if leaky:
      self.relu = nn.LeakyReLU(negative_slope=0.2)
    else:
      self.relu = nn.ReLU()

  def forward(self, x, **kwargs):
    return self.relu(self.norm(self.dropout(self.conv(x))))

class UNet1D(nn.Module):
  '''
  UNet model for 1D inputs
  (cite: ``https://arxiv.org/pdf/1505.04597.pdf``)

  Arguments
    input_channels (int): input channel size
    output_channels (int): output channel size (or the number of output features to be predicted)
    max_depth (int, optional): depth of the UNet (default: ``5``).
    kernel_size (int, optional): size of the kernel for each convolution (default: ``None``)
    stride (int, optional): stride of the convolution layers (default: ``None``)

  Shape
    Input: :math:`(N, C_{in}, L_{in})`
    Output: :math:`(N, C_{out}, L_{out})` where
      .. math::
        assert L_{in} >= 2^{max_depth - 1}
        L_{out} = L_{in}
        C_{out} = output_channels

  Inputs
    x (torch.Tensor): speech signal in form of a 3D Tensor

  Outputs
    x (torch.Tensor): input transformed to a lower frequency
      latent vector

  '''
  def __init__(self, input_channels, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
    super(UNet1D, self).__init__()
    self.pre_downsampling_conv = nn.ModuleList([])
    self.conv1 = nn.ModuleList([])
    self.conv2 = nn.ModuleList([])
    self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
    self.max_depth = max_depth
    self.groups = groups

    ## pre-downsampling
    self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.pre_downsampling_conv.append(ConvNormRelu(output_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    for i in range(self.max_depth):
      self.conv1.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=True,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    for i in range(self.max_depth):
      self.conv2.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=False,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, return_bottleneck=False):
    input_size = x.shape[-1]
    assert input_size/(2**(self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(input_size, 2**(self.max_depth - 1))
    #assert np.log2(input_size) == int(np.log2(input_size)), 'Input size is {}. It must be a power of 2.'.format(input_size)
    assert num_powers_of_two(input_size) >= self.max_depth, 'Input size is {}. It must be a multiple of 2^(max_depth) = 2^{} = {}'.format(input_size, self.max_depth, 2**self.max_depth)

    x = nn.Sequential(*self.pre_downsampling_conv)(x)

    residuals = []
    residuals.append(x)
    for i, conv1 in enumerate(self.conv1):
      x = conv1(x)
      if i < self.max_depth - 1:
        residuals.append(x)

    bn = x
    for i, conv2 in enumerate(self.conv2):
      x = self.upconv(x) + residuals[self.max_depth - i - 1]
      x = conv2(x)

    if return_bottleneck:
      return x, bn
    else:
      return x

class AudioEncoder(nn.Module):
  '''
  input_shape:  (N, C, time, frequency)
  output_shape: (N, 256, output_feats)
  '''
  def __init__(self, output_feats=64, input_channels=1, kernel_size=None, stride=None, p=0, groups=1):
    super(AudioEncoder, self).__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(128, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=(3,8), stride=1, p=p, groups=groups))

    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x

class PoseEncoder(nn.Module):
  '''
  input_shape:  (N, time, pose_features: 104) #changed to 96?
  output_shape: (N, 256, time)
  '''
  def __init__(self, output_feats=64, input_channels=96, kernel_size=None, stride=None, p=0, groups=1):
    super(PoseEncoder, self).__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))



    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x

    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

class PoseStyleEncoder(nn.Module):
  '''
  input_shape:  (N, time, pose_features: 104) #changed to 96?
  output_shape: (N, 256, t)
  '''
  def __init__(self, output_feats=64, input_channels=96, kernel_size=None, stride=None, p=0, groups=1, num_speakers=4):
    super().__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(256, num_speakers, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))



    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.mean(-1)
    x = x.squeeze(dim=-1)
    return x
    
class PoseDecoder(nn.Module):
  '''
  input_shape:  (N, channels, time)
  output_shape: (N, 256, time)
  '''
  def __init__(self, input_channels=256, style_dim=10, num_clusters=8, out_feats=96, kernel_size=None, stride=None, p=0):
    super().__init__()
    self.num_clusters = num_clusters
    self.style_dim = style_dim
    self.pose_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(input_channels+style_dim,
                                                                   input_channels,
                                                                   type='1d', leaky=True, downsample=False,
                                                                   p=p, groups=num_clusters)
                                                      for i in range(4)]))
    self.pose_logits = nn.Conv1d(input_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

  def forward(self, x, **kwargs):
    style = x.view(x.shape[0], -1, self.num_clusters, x.shape[-1])[:, -self.style_dim:]
    for i, model in enumerate(self.pose_decoder):
      #x = torch.split(x, int(x.shape[1]/self.num_clusters), dim=1)
      #x = torch.cat([torch.cat([x_, kwargs['style']], dim=1) for x_ in x], dim=1)
      x = model(x)
      if i < len(self.pose_decoder) - 1: ## ignore last layer
        x = x.view(x.shape[0], -1, self.num_clusters, x.shape[-1])
        x = torch.cat([x, style], dim=1).view(x.shape[0], -1, x.shape[-1])
    return self.pose_logits(x)

class StyleDecoder(nn.Module):
  '''
  input_shape:  (N, channels, time)
  output_shape: (N, 256, time)
  '''
  def __init__(self, input_channels=256, num_clusters=10, out_feats=96, kernel_size=None, stride=None, p=0):
    super().__init__()
    self.num_clusters = num_clusters
    self.pose_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(input_channels,
                                                                   input_channels,
                                                                   type='1d', leaky=True, downsample=False,
                                                                   p=p, groups=num_clusters)
                                                      for i in range(2)]))
    self.pose_logits = nn.Conv1d(input_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

  def forward(self, x, **kwargs):
    x = self.pose_decoder(x)
    return self.pose_logits(x)


#TODO Unify Encoders via input_channel size?
class TextEncoder1D(nn.Module):
  '''
  input_shape:  (N, time, text_features: 300)
  output_shape: (N, 256, time)
  '''
  def __init__(self, output_feats=64, input_channels=300, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.conv = nn.ModuleList([])

    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, time_steps=None, **kwargs):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x

class MixGANDecoder(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters

      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))
      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

      self.labels_cap_soft = None

    def index_select_outputs(self, x, labels):
      '''
      x: (B, num_clusters*out_feats, T)
      labels: (B, T, num_clusters)
      '''
      x = x.transpose(2, 1)
      x = x.view(x.shape[0], x.shape[1], self.num_clusters, -1)
      labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling

      x = (x * labels.unsqueeze(-1)).sum(dim=-2)
      return x

    def forward(self, x, labels, **kwargs):
      internal_losses = []

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
 
      return x, internal_losses

#Positional Encoding missing in vanilla Transformer
#source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class PositionalEncoding(nn.Module):
    def __init__(self, input_channels=300, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_channels, 2).float() * (-math.log(10000.0) / input_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FramesPositionalEncoding(nn.Module):
    def __init__(self, input_channels=300, dropout=0.1, max_len=5000, batch_size = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_channels, 2).float() * (-math.log(10000.0) / input_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x, text_duration, train):
        text_duration_long = text_duration.long()
        sample_flag = 0 if x.shape[1] != text_duration.shape[0] else 1
        if sample_flag:
            with torch.no_grad():
                for i, interval in enumerate(text_duration_long):
                    new_pos = 0
                    for j, word_dur in enumerate(interval):
                        try:
                            x[new_pos: new_pos + word_dur, i,:] += self.pe[0: word_dur,:]
                            new_pos += word_dur
                        except:
                            pdb.post_mortem()
        else:
            text_duration_long = text_duration.long()
            text_collapsed = text_duration_long.reshape(-1)
            new_pos = 0
            for i, word_dur in enumerate(text_collapsed):
                x[new_pos:new_pos + word_dur,0,:] += self.pe[0: word_dur,:]
                new_pos += word_dur
        return self.dropout(x)


class RepeatWordPositionalEncoding(nn.Module): #word repeats
    def __init__(self, input_channels=300, dropout=0.1, max_len=5000, batch_size = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_channels, 2).float() * (-math.log(10000.0) / input_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x, text_duration, train):
        text_duration_long = text_duration.long()
        if train:
            with torch.no_grad():
                for i, interval in enumerate(text_duration_long):
                    new_pos = 0
                    for j, word_dur in enumerate(interval):
                        x[new_pos: new_pos + word_dur, i,:] += self.pe[j: j+1,:]
                        new_pos += word_dur
        else:
            text_duration_long = text_duration.long()
            text_collapsed = text_duration_long.reshape(-1)
            new_pos = 0
            for i, word_dur in enumerate(text_collapsed):
                x[new_pos:new_pos + word_dur,0,:] += self.pe[0: word_dur,:]
                new_pos += word_dur
        return self.dropout(x)


class TransfomerEncoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()
    ## Linear Layers
    self.tlinear_enc = nn.Linear(in_channels, E)
    self.ptlinear_enc = nn.Linear(out_feats, E)
    self.linear_decoder = nn.Linear(E, out_feats)

    ## Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.ninp = E
    self.pos_encoder = PositionalEncoding(self.ninp, p)

    encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid)
    encoder_norm = torch.nn.LayerNorm(self.ninp)
    self.transformer_text_encoder = nn.TransformerEncoder(encoder_layers, self.nhid, encoder_norm) # Norm

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def _generate_source_key_padding_mask(self, token_count):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count)
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = 0
    return mask.bool().to(token_count.device)

  def forward(self, x, y, input_repeat = 0, output_repeat=0, **kwargs):
    #pdb.set_trace()
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'])
    text_duration = kwargs['text/token_duration']
    if src_key_padding_mask.shape[1] != x.shape[1]:
      src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
      text_duration = text_duration.view(x.shape[0], x.shape[1])
    memory = x.transpose(0, 1)
    memory = self.tlinear_enc(memory)
    memory = self.pos_encoder(memory)
    memory = self.transformer_text_encoder(memory, src_key_padding_mask=src_key_padding_mask)
    if output_repeat:
      assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
      batch_memory_list = []
      for b in range(memory.shape[1]): # batch
        memory_list = []
        for i in range(memory.shape[0]): # word
            repeats = int(text_duration[b, i].item())
            if (repeats != 0):
                memory_list_ = [memory[i, b:b+1].repeat(int(text_duration[b, i].item()), 1, 1) ]
                memory_list.append(torch.cat(memory_list_, dim=0))
        sec_memory = torch.cat(memory_list, dim=0)
        batch_memory_list.append(sec_memory)
      final_memory = torch.cat(batch_memory_list, dim=1)
    return final_memory.transpose(0, 1) ## (B, time, channels)

class TransfomerEncoder2(TransfomerEncoder):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()

  def forward(self, x, y, input_repeat = 1,  output_repeat=1, **kwargs): #repeated text, kwargs token_count, duration not given
    #src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count']) #not needed
    # text_duration = kwargs['text/token_duration'] #not given
    # if src_key_padding_mask.shape[1] != x.shape[1]:
    #   src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
    #   text_duration = text_duration.view(x.shape[0], x.shape[1])
    memory = x.transpose(0, 1)
    memory = self.tlinear_enc(memory)
    memory = self.pos_encoder(memory)
    #memory = self.transformer_text_encoder(memory, src_key_padding_mask=src_key_padding_mask)
    # if repeat_text:
    #   #assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    #   memory_list = []
    #   for b in range(memory.shape[1]):
    #     memory_list_ = [memory[i, b:b+1].repeat(int(text_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
    #     memory_list.append(torch.cat(memory_list_, dim=0))
    #   memory = torch.cat(memory_list, dim=1)
    return memory.transpose(0, 1) ## (B, time, channels)


class TransfomerEncoder_WordPOS(TransfomerEncoder):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=2, **kwargs):
    super().__init__()
    ## Linear Layers
    self.tlinear_enc = nn.Linear(in_channels, E)
    self.ptlinear_enc = nn.Linear(out_feats, E)
    self.linear_decoder = nn.Linear(E, out_feats)

    ## Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.ninp = E
    self.pos_encoder = PositionalEncoding(self.ninp, p)
    self.frames_pos_encoder = FramesPositionalEncoding(self.ninp, p)
    encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid)
    encoder_norm = torch.nn.LayerNorm(self.ninp)
    self.transformer_text_encoder = nn.TransformerEncoder(encoder_layers, self.nhid, encoder_norm) # Norm

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def _generate_source_key_padding_mask(self, token_count):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count)
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = 0
    return mask.bool().to(token_count.device)


  def forward(self, x, y, input_repeat = 1,  output_repeat=1, **kwargs):
    text_duration = kwargs['text/token_duration']
    memory = x.transpose(0, 1)
    memory = self.tlinear_enc(memory)
    train = True
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'])
    if src_key_padding_mask.shape[0] != x.shape[0]:
      train = False
    memory = self.frames_pos_encoder(memory, text_duration, train)
    memory = self.transformer_text_encoder(memory) #mask unneeded for source
    #pdb.set_trace()
    # if repeat_text:
    #   assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    #   memory_list = []
    #   for b in range(memory.shape[1]):
    #     memory_list_ = [memory[i, b:b+1].repeat(int(text_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
    #     memory_list.append(torch.cat(memory_list_, dim=0))
    #   memory = torch.cat(memory_list, dim=1)
    return memory.transpose(0, 1) ## (B, time, channels)



class TransfomerEncoder_Multi(TransfomerEncoder_WordPOS): #Word Level + Frame Level Pos Encoding
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()
    self.word_pos_encoder = RepeatWordPositionalEncoding(self.ninp, p)


  def forward(self, x, y, input_repeat = 1,  output_repeat=1, **kwargs):
    text_duration = kwargs['text/token_duration']
    memory = x.transpose(0, 1)
    memory = self.tlinear_enc(memory)
    train = True
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'])
    if src_key_padding_mask.shape[0] != x.shape[0]:
      train = False
    memory = self.word_pos_encoder(memory, text_duration, train)
    memory = self.transformer_text_encoder(memory)
    memory = self.frames_pos_encoder(memory, text_duration, train)
    memory = self.transformer_text_encoder(memory) #mask unneeded for source


    #pdb.set_trace()
    # if repeat_text:
    #   assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    #   memory_list = []
    #   for b in range(memory.shape[1]):
    #     memory_list_ = [memory[i, b:b+1].repeat(int(text_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
    #     memory_list.append(torch.cat(memory_list_, dim=0))
    #   memory = torch.cat(memory_list, dim=1)
    return memory.transpose(0, 1) ## (B, time, channels)



class TransfomerDecoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()
    ## Linear Layers
    self.plinear_enc = nn.Linear(out_feats, E)
    self.linear_decoder = nn.Linear(E, out_feats)
    self.decoder_emb = nn.Linear(E, out_feats)

    ## Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.ninp = E
    self.pos_encoder = PositionalEncoding(self.ninp, p)

    #Decoder
    decoder_layers = nn.TransformerDecoderLayer(E, self.nhead, self.nhid)
    decoder_norm = torch.nn.LayerNorm(self.ninp)
    self.transformer_text_decoder = nn.TransformerDecoder(decoder_layers, self.nhid, decoder_norm) # Norm\

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def _generate_source_key_padding_mask(self, token_count):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count)
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = 0
    return mask.bool().to(token_count.device)

  def forward(self, memory, y, time_steps=None, **kwargs):
    tgt_mask = self._generate_square_subsequent_mask(y.shape[1]).to(y.device).double()
    if time_steps is None:
      time_steps = y.shape[1]
    memory = memory.transpose(0, 1)
    y = y.transpose(0, 1)

    if self.training:
      y = self.plinear_enc(y)
      output = self.transformer_text_decoder(y, memory, tgt_mask = tgt_mask)
    else:
      batch_size = y.shape[1]
      output = torch.zeros(time_steps+1, batch_size, self.ninp).double().to(y.device)
      for t in range(1, time_steps+1):
        tgt_emb = output[:t]
        decoder_output = self.transformer_text_decoder(tgt_emb, memory)
        output[t] = decoder_output[-1]
        #output[1:t+1] = decoder_output ## output of decoder updated at every step
      output = output[1:] ## removing the starting zero
    output = self.linear_decoder(output)
    return output.transpose(0, 1)

class TransfomerDecoderRand(TransfomerDecoder):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__(time_steps, in_channels, out_feats, p, E, nhead, nhid, **kwargs)

  def forward(self, memory, y, time_steps=None, **kwargs):
    tgt_mask = self._generate_square_subsequent_mask(y.shape[1]).to(y.device).double()
    if time_steps is None:
      time_steps = y.shape[1]
    memory = memory.transpose(0, 1)
    y = y.transpose(0, 1)

    y = torch.rand_like(y)
    y = self.plinear_enc(y)
    output = self.transformer_text_decoder(y, memory, tgt_mask = tgt_mask)
    output = self.linear_decoder(output)
    return output.transpose(0, 1)


class TextEncoderTransformer_d(nn.Module):
      '''
      input_shape:  (N, time, text_features: 300)
      output_shape: (N, 256, output_feats)
      '''
      def __init__(self, ntokens = 30, output_feats=64, input_channels=300, kernel_size=None, stride=None, p=0):
        super().__init__()
        self.n_heads = 12
        self.n_layers = 3
        self.ntoken = ntokens#TODO: needs to be found
        self.input_channels = input_channels
        self.pos_encoder = PositionalEncoding(input_channels)
        self.encoder_layer = nn.TransformerEncoderLayer(input_channels, self.n_heads, 256)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.n_layers)
        self.encoder = nn.Embedding(self.ntoken, input_channels) #token
        self.conv = ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                                 kernel_size=kernel_size, stride=stride, p=p)


      def forward(self, x, time_steps=None):
        if time_steps is None:
          time_steps = x.shape[-2]

        x = self.encoder(x) * math.sqrt(self.input_channels)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x.transpose(1, 0)).transpose(1, 0).transpose(2, 1)
        x = self.conv(x)
        return x

class TextEncoderTransformer(nn.Module):
  '''
  input_shape:  (N, time, text_features: 300)
  output_shape: (N, 256, output_feats)
  '''
  def __init__(self, output_feats=64, input_channels=300, kernel_size=None, stride=None, p=0):
    super().__init__()
    self.n_heads = 12
    self.n_layers = 3
    self.encoder_layer = nn.TransformerEncoderLayer(input_channels, self.n_heads, 256)
    self.encoder = nn.TransformerEncoder(self.encoder_layer, self.n_layers)

    self.conv = ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                             kernel_size=kernel_size, stride=stride, p=p)

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2]
    x = self.encoder(x.transpose(1, 0)).transpose(1, 0).transpose(2, 1)
    x = self.conv(x)
    return x

    #TODO Unify Encoders via input_channel size?

class BertEncoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.linear = torch.nn.Linear(768, out_feats)

  def _generate_source_key_padding_mask(self, token_count, mask_val=0):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count) - mask_val
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = mask_val
    return mask.bool().to(token_count.device)

  def output_repeat_text(self, memory, token_duration):
    memory_list = []
    for b in range(memory.shape[1]):
      memory_list_ = [memory[i, b:b+1].repeat(int(token_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
      memory_list.append(torch.cat(memory_list_, dim=0))
    memory = torch.cat(memory_list, dim=1)
    return memory

  def chunk(self, x, pad, max_len=512):
    x_len = x.shape[-1]
    batch = (x_len - 1) // max_len + 1
    if batch > 1:
      new_len = max_len * batch
      x = torch.cat([x, torch.zeros(1, new_len-x_len).double().to(x.device)], dim=-1)
      pad = torch.cat([pad, torch.zeros(1, new_len-x_len).bool().to(x.device)], dim=-1)
      x = x.view(batch, -1)
      pad = pad.view(batch, -1)
    
    return x, pad, x_len, batch
    
  def forward(self, x, y, input_repeat = 0, output_repeat=0, **kwargs):  
    token_type_ids = None
    if len(x.shape) == 3:
      sample_flag = True
    else:
      sample_flag = False

    ## Create Masks
    assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    token_duration = kwargs['text/token_duration']
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'], mask_val=1)
    #if src_key_padding_mask.shape[1] != x.shape[1]:
    if sample_flag:
      x = x.view(1, -1)
      src_key_padding_mask = src_key_padding_mask.view(1, -1)
      x, src_key_padding_mask, orig_len, batch = self.chunk(x, src_key_padding_mask)

      #src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
      
    memory, pooled_output = self.bert(x.long(), token_type_ids, src_key_padding_mask.long())
    
    memory = self.linear(memory)

    if sample_flag:
      memory = memory.view(1, -1, memory.shape[-1])[:, :orig_len]
      token_duration = token_duration.view(memory.shape[0], memory.shape[1])[:, :orig_len]

    if 'pos_encoder' in kwargs: ## add positional embedding before repeating -- Useful is used in conjunction with another transformer
      memory = kwargs['pos_encoder'](memory.transpose(1, 0)).transpose(1, 0) ## needs input in the form of (T, B, C)
      
    if output_repeat:
      memory = self.output_repeat_text(memory.transpose(1, 0), token_duration).transpose(1, 0)

    return memory.transpose(-1, -2)

  def forward_archive(self, x, y, input_repeat = 0, output_repeat=0, **kwargs):
    token_type_ids = None
    if len(x.shape) == 3:
      sample_flag = True
      x = x.squeeze(0)
    else:
      sample_flag = False

    ## Create Masks
    assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    token_duration = kwargs['text/token_duration']
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'], mask_val=1)
    #if src_key_padding_mask.shape[1] != x.shape[1]:
    # if sample_flag:
    #   x = x.view(1, -1)
    #   src_key_padding_mask = src_key_padding_mask.view(1, -1)
    #   x, src_key_padding_mask, orig_len, batch = self.chunk(x, src_key_padding_mask)

      #src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
    memory, pooled_output = self.bert(x.long(), token_type_ids, src_key_padding_mask.long())
    memory = self.linear(memory)

    # if sample_flag:
    #   memory = memory.view(1, -1, memory.shape[-1])[:, :orig_len]
    #   token_duration = token_duration.view(memory.shape[0], memory.shape[1])[:, :orig_len]

    if 'pos_encoder' in kwargs: ## add positional embedding before repeating -- Useful is used in conjunction with another transformer
      memory = kwargs['pos_encoder'](memory.transpose(1, 0)).transpose(1, 0) ## needs input in the form of (T, B, C)
      
    if output_repeat:
      memory = self.output_repeat_text(memory.transpose(1, 0), token_duration).transpose(1, 0)

    if sample_flag:
      memory = memory.view(1, -1, memory.shape[-1])
    
    return memory.transpose(-1, -2)

  
class MultiScaleBertEncoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, p = 0, E = 256, nhead=8, nhid=256, **kwargs):
    super().__init__()
    self.word_encoder = BertEncoder(out_feats=out_feats)

    ## Frame Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.ninp = out_feats
    self.pos_encoder = PositionalEncoding(self.ninp, p)
    self.frame_pos_encoder = FramesPositionalEncoding(input_channels=E, dropout=0)

    encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid)
    encoder_norm = torch.nn.LayerNorm(self.ninp)
    self.frame_encoder = nn.TransformerEncoder(encoder_layers, self.nhid, encoder_norm) # Norm

  def forward(self, x, y, input_repeat=0, output_repeat=1, **kwargs):
    if kwargs['description'] == 'train':
      is_train = True
    else:
      is_train = False
    memory = self.word_encoder(x, y, input_repeat=0, output_repeat=1, pos_encoder=self.pos_encoder, **kwargs).transpose(-1, -2) ## (B, T) -> (B, T, C)
    memory = self.frame_pos_encoder(memory.transpose(1, 0), kwargs['text/token_duration'], is_train) # (T, B, C) as input -> (T, B, C)
    memory = self.frame_encoder(memory)
    return memory.transpose(1, 0).transpose(-1, -2) # (B, C, T)

class MultimodalTransformerFusion(nn.Module):
  '''
  tgt: audio signal (T, B, C)
  src: text signal (L, B, C), if input_repeat == 0 => L!=T and if input_repeat == 1 => L==T
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, p = 0, E = 256, nhead=8, nhid=256, nlayer=2,**kwargs):
    super().__init__()
    ## Frame Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.nlayer = nlayer
    self.ninp = out_feats

    decoder_layers = nn.TransformerDecoderLayer(self.ninp, self.nhead, self.nhid)
    decoder_norm = torch.nn.LayerNorm(self.ninp)
    self.memory_decoder = nn.TransformerDecoder(decoder_layers, self.nlayer, decoder_norm) # Norm

  def _generate_source_key_padding_mask(self, token_count, mask_val=0):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count) - mask_val
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = mask_val
    return mask.bool().to(token_count.device)

  def _generate_source_mask(self, token_duration, tgt_len, bsz, input_repeat):
    if input_repeat == 0:
      mask = torch.ones(bsz*self.nhead, tgt_len, token_duration.shape[-1]) # (B, T, L)
    else:
      mask = torch.ones(bsz*self.nhead, tgt_len, tgt_len) # (B, T, T)
    for b in range(token_duration.shape[0]):
      pos = 0
      for i in range(token_duration.shape[1]):
        duration = int(token_duration[b, i].item())
        if input_repeat == 0:
          mask[b*self.nhead:(b+1)*self.nhead, pos:pos+duration, i] = 0
        else:
          mask[b*self.nhead:(b+1)*self.nhead, pos:pos+duration, pos:pos+duration] = 0
        pos = pos + duration
    #mask = mask.float().masked_fill(mask==1, float('-inf')).masked_fill(mask==0, float(0.0)).to(token_duration.device)
    #return mask
    return mask.bool().to(token_duration.device)


  def output_repeat_text(self, memory, token_duration):
    memory_list = []
    for b in range(memory.shape[1]):
      memory_list_ = [memory[i, b:b+1].repeat(int(token_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
      memory_list.append(torch.cat(memory_list_, dim=0))
    memory = torch.cat(memory_list, dim=1)
    return memory

  '''
  tgt: (B, C, T) -> (T, B, C)
  memory: (B, C, L) -> (L, B, C)
  '''
  def forward(self, tgt, memory, y, input_repeat=0, output_repeat=1, src_mask=True, query_text=False, **kwargs):
    tgt = tgt.permute(2, 0, 1)
    memory = memory.permute(2, 0, 1)
    if kwargs['description'] == 'train':
      is_train = True
    else:
      is_train = False

    token_duration = kwargs['text/token_duration']
    token_count = kwargs['text/token_count']
    if token_duration.shape[0] != tgt.shape[1]: ## sample_loop
      token_duration = token_duration.view(1, -1)
      sample_flag = True
    else:
      sample_flag = False
      
    if src_mask:
      src_mask = self._generate_source_mask(token_duration, tgt.shape[0], tgt.shape[1], input_repeat)
    else:
      src_mask = None
    if input_repeat == 0:
      src_key_padding_mask = self._generate_source_key_padding_mask(token_count)
      if sample_flag:
        src_key_padding_mask = src_key_padding_mask.view(1, -1)
    else:
      src_key_padding_mask = None

    if not query_text:
      ## memory(~key and value) is text, tgt (~query) is audio
      memory = self.memory_decoder(tgt, memory, memory_key_padding_mask=src_key_padding_mask, memory_mask=src_mask)
    else:
      memory = self.memory_decoder(memory, tgt, tgt_key_padding_mask=src_key_padding_mask, tgt_mask=src_mask)
  
    return memory.transpose(1, 0).transpose(-1, -2) # (B, C, T)

  
class Transpose(nn.Module):
  def __init__(self, idx):
    super().__init__()
    self.param = torch.nn.Parameter(torch.ones(1))
    self.idx = idx

  def forward(self, x, *args, **kwargs):
    return x.transpose(*self.idx)
  
class AudioEncoder1D(nn.Module):
  '''
  input_shape:  (N, time, audio_features: 128)
  output_shape: (N, 256, output_feats)
  '''
  def __init__(self, output_feats=64, input_channels=128, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, time_steps=None):
    #x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x


        ## deprecated, but kept only for older models
        # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
        ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

class LatentEncoder(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels=2, p=0):
    super().__init__()
    enc1 = nn.ModuleList([ConvNormRelu(in_channels, hidden_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=1)
                          for i in range(1)])
    enc2 = nn.ModuleList([ConvNormRelu(hidden_channels, hidden_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=1)
                          for i in range(2)])
    enc3 = nn.ModuleList([ConvNormRelu(hidden_channels, out_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=1)
                          for i in range(1)])
    self.enc = nn.Sequential(*enc1, *enc2, *enc3)

  def forward(self, x):
    x = self.enc(x)
    return x

        
class VQLayer(nn.Module):
  '''
  VQ Layer without Stop gradient
  '''
  def __init__(self, num_embeddings=8, num_features=96, weight=None):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.emb = nn.Embedding(self.num_embeddings, self.num_features, _weight=weight)

  def forward(self, x):
    x = x.transpose(-1, -2) ## (B, T, num_features)
    x = x.view(x.shape[0], x.shape[1], x.shape[2], 1) ## (B, T, num_features)
    centers = self.emb.weight.transpose(1, 0) ## (num_features, num_embeddings)
    centers = centers.view(1, 1, centers.shape[0], centers.shape[1]) ## (1, 1, num_features, num_embeddings)
    dist = ((x-centers)**2).sum(dim=-2) ## (B, T, num_embeddings)
    idxs = torch.argmin(dist, dim=-1) ## (B, T)

    return self.emb(idxs).transpose(-1, -2), dist

class VQLayerSG(nn.Module):
  '''
  VQ layer with stop gradient
  '''
  def __init__(self, num_embeddings=8, num_features=96, weight=None):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.emb = nn.Embedding(self.num_embeddings, self.num_features, _weight=weight)

  def get_dist(self, x, y):
    return ((x - y)**2).mean(dim=-2)

  def get_min_dist(self, dist):
    return dist.min(dim=-1)[0].mean()

  def forward(self, x):
    x = x.transpose(-1, -2) ## (B, T, num_features)
    x = x.view(x.shape[0], x.shape[1], x.shape[2], 1) ## (B, T, num_features)
    centers = self.emb.weight.transpose(1, 0) ## (num_features, num_embeddings)
    centers = centers.view(1, 1, centers.shape[0], centers.shape[1]) ## (1, 1, num_features, num_embeddings)
    dist = self.get_dist(x, centers.detach()) ## (B, T, num_embeddings)
    dist, idxs = dist.min(dim=-1) ## (B, T)

    internal_losses = []
    internal_losses.append(dist.mean())
    internal_losses.append(self.get_min_dist(self.get_dist(x.detach(), centers)))

    ## get the output
    idxs_shape = list(idxs.shape)
    idxs = idxs.view(-1)
    out = torch.index_select(self.emb.weight.detach(), dim=0, index=idxs)
    out_shape = idxs_shape + [out.shape[-1]]
    out = out.view(*out_shape)
    return out.transpose(-1, -2), internal_losses


class ClusterClassify(nn.Module):
  '''
  input_shape: (B, C, T)
  output_shape: (B, num_clusters, T)
  '''
  def __init__(self, num_clusters=8, kernel_size=None, stride=None, p=0, groups=1, input_channels=256):
    super().__init__()
    self.conv = nn.ModuleList()
    self.conv.append(ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv += nn.ModuleList([ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                             kernel_size=kernel_size, stride=stride, p=p, groups=groups) for i in range(5)])

    self.logits = nn.Conv1d(256*groups, num_clusters*groups, kernel_size=1, stride=1, groups=groups)

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    x = self.logits(x)
    return x

class ClusterClassifyGAN(nn.Module):
  '''
  input_shape: (B, C, T)
  output_shape: (B, T, num_clusters)
  '''
  def __init__(self, num_clusters=8, kernel_size=None, stride=None, p=0, groups=1, input_channels=256):
    super().__init__()
    self.conv = nn.ModuleList()
    self.conv.append(ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv += nn.ModuleList([ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                             kernel_size=kernel_size, stride=stride, p=p, groups=groups) for i in range(5)])

    self.logits = nn.Conv1d(256*groups, num_clusters*groups, kernel_size=1, stride=1, groups=groups)

  def forward(self, x, y, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    x = self.logits(x)
    return x.transpose(-1, -2), []


class Confidence(nn.Module):
  '''
  0 < confidence <= 1
  '''
  def __init__(self, beta=0.1, epsilon=1e-8):
    super().__init__()
    self.beta = beta
    self.epsilon = epsilon

  def forward(self, y, y_cap, confidence):
    if isinstance(confidence, int):
      confidence = torch.ones_like(y)
    sigma = self.get_sigma(confidence)
    P_YCAP_Y = self.p_ycap_y(y, y_cap, sigma)
    sigma_ycap = self.get_sigma(P_YCAP_Y)
    return self.get_entropy(sigma_ycap)

  def p_ycap_y(self, y, y_cap, sigma):
    diff = -(y-y_cap)**2
    diff_normalized = diff/(2*sigma**2)
    prob = torch.exp(diff_normalized)
    prob_normalized = prob*(1/(2*math.pi*sigma))
    return prob_normalized

  def get_sigma(self, confidence):
    mask = (confidence < self.epsilon).double()
    confidence = (1 - mask) * confidence + torch.ones_like(confidence)*self.epsilon*mask
    sigma = 1/(2*math.pi*confidence)
    return sigma

  ## entropy of a guassian
  def get_entropy(self, sigma):
    return 0.5*(torch.log(2*math.pi*math.e*(sigma**2)))*self.beta

class Repeat(nn.Module):
  def __init__(self, repeat, dim=-1):
    super().__init__()
    self.dim = dim
    self.repeat = repeat
    #self.temp = torch.nn.Parameter(torch.zeros(1))

  def forward(self, x):
    return x.repeat_interleave(self.repeat, self.dim)


class BatchGroup(nn.Module):
  '''
  Group conv networks to run in parallel
  models: list of instantiated models

  Inputs:
    x: list of list of inputs; x[group][batch], len(x) == groups, and len(x[0]) == batches
    labels: uses these labels to give a soft attention on the outputs. labels[batch], len(labels) == batches
            if labels is None, return a list of outputs
    transpose: if true, model needs a transpose of the input
  '''
  def __init__(self, models, groups=1):
    super().__init__()
    if not isinstance(models, list):
      models = [models]
    self.models = nn.ModuleList(models)
    self.groups = groups

  def index_select_outputs(self, x, labels):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], self.groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x

  def forward(self, x, labels=None, transpose=True, **kwargs):
    if not isinstance(x, list):
      raise 'x must be a list'
    if not isinstance(x[0], list):
      raise 'x must be a list of lists'
    if labels is not None:
      assert isinstance(labels, list), 'labels must be a list'

    groups = len(x)
    assert self.groups == groups, 'input groups should be the same as defined groups'
    batches = len(x[0])

    x = [torch.cat(x_, dim=0) for x_ in x] # batch
    x = torch.cat(x, dim=1)  # group

    if transpose:
      x = x.transpose(-1, -2)
    for model in self.models:
      if kwargs:
        x = model(x, **kwargs)
      else:
        x = model(x)

    is_tuple = isinstance(x, tuple)
    if labels is not None:
      assert not is_tuple, 'labels is not None does not work with is_tuple=True'
      labels = torch.cat(labels, dim=0) # batch
      x = [self.index_select_outputs(x, labels).transpose(-1, -2)]
    else: # separate the groups
      if is_tuple:
        channels = [int(x[i].shape[1]/groups) for i in range(len(x))]
        x = [torch.split(x_, channels[i], dim=1) for i, x_ in enumerate(x)]
        #x = list(zip(*[torch.split(x_, channels[i], dim=1) for i, x_ in enumerate(x)]))
        #x = [tuple([x_[:, start*channels[i]:(start+1)*channels[i]] for i, x_ in enumerate(x)]) for start in range(groups)]
      else:
        channels = int(x.shape[1]/groups)
        x = list(torch.split(x, channels, dim=1))
        #x = [x[:, start*channels:(start+1)*channels] for start in range(groups)]

    if is_tuple:
      channels = int(x[0][0].shape[0]/batches)
      x = tuple([[torch.split(x__, channels, dim=0) for x__ in x_] for x_ in x])
      #x = [[tuple([x__[start*channels:(start+1)*channels] for x__ in x_]) for start in range(batches)] for x_ in x]
    else:
      channels = int(x[0].shape[0]/batches)
      x = [list(torch.split(x_, channels, dim=0)) for x_ in x]
      #x = [[x_[start*channels:(start+1)*channels] for start in range(batches)] for x_ in x]
    return x


class Group(nn.Module):
  '''
  Group conv networks to run in parallel
  models: list of instantiated models
  groups: groups of inputs
  dim: if dim=0, use batch a set of inputs along batch dimension (group=1 always)
       elif dim=1, combine the channel dimension (group=num_inputs)

  Inputs:
    x: list of inputs
    labels: uses these labels to give a soft attention on the outputs. Use only with dim=1.
            if labels is None, return a list of outputs
    transpose: if true, model needs a transpose of the input
  '''
  def __init__(self, models, groups=1, dim=1):
    super().__init__()
    if not isinstance(models, list):
      models = [models]
    self.models = nn.ModuleList(models)
    self.groups = groups
    self.dim = dim

  def index_select_outputs(self, x, labels):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], self.groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x

  def forward(self, x, labels=None, transpose=True, **kwargs):
    if self.dim == 0:
      self.groups = len(x)
    if isinstance(x, list):
      x = torch.cat(x, dim=self.dim) ## concatenate along channels
    if transpose:
      x = x.transpose(-1, -2)
    for model in self.models:
      if kwargs:
        x = model(x, **kwargs)
      else:
        x = model(x)
    if labels is not None:
      x = self.index_select_outputs(x, labels).transpose(-1, -2) ## only for dim=1
      return x
    else:
      channels = int(x.shape[self.dim]/self.groups)
      dim = self.dim % len(x.shape)
      if dim == 2:
        x = [x[:, :, start*channels:(start+1)*channels] for start in range(self.groups)]
      elif dim == 1:
        x = [x[:, start*channels:(start+1)*channels] for start in range(self.groups)]
      elif dim == 0:
        x = [x[start*channels:(start+1)*channels] for start in range(self.groups)]
      return x

class EmbLin(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def forward(self, x, mode='lin'):
    if mode == 'lin':
      return x.matmul(self.emb.weight)
    elif mode == 'emb':
      return self.emb(x)

class EmbLin2(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, groups=1):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.groups = groups
    self.emb = nn.Embedding(num_embeddings, embedding_dim*self.groups)

  def forward(self, x, mode='lin'):
    if mode == 'lin':
      return x.matmul(self.emb.weight)
    elif mode == 'emb':
      return self.emb(x)


def get_roll_value(x, num_styles):
  if isinstance(x, list):
    shape = x[0].shape[0]
  else:
    shape = x.shape[0]

  roll_value = torch.arange(0, shape)
  if num_styles > 1:
    roll_value = roll_value[roll_value%num_styles!=0]
  else:
    roll_value = roll_value[1:]
  roll_value = roll_value[torch.randint(0, len(roll_value), size=(1,))].item()
  return roll_value

def roll(x, roll_value):
  if isinstance(x, list):
    return [torch.roll(x_, roll_value, dims=0) for x_ in x]
  else:
    return torch.roll(x, roll_value, dims=0)

class Style(nn.Module):
  '''
  input_shape: (B, )
  output_shape: (B, )
  '''
  def __init__(self, num_speakers=1):
    self.style_emb = nn.Embedding(num_embeddings=num_speakers, embedding_dim=256)

  def forward(self, x):
    pass

class Curriculum():
  def __init__(self, start, end, num_iters):
    self.start = start
    self.end = end
    self.num_iters = num_iters
    self.iters = 0
    self.diff = (end-start)/num_iters
    self.value = start

  def step(self, flag=True):
    if flag:
      value_temp = self.value
      if self.iters < self.num_iters:
        self.value += self.diff
        self.iters += 1
        return value_temp
      else:
        return self.end
    else:
      return self.value

