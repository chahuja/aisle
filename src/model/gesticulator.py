import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

GestLate_D = Speech2Gesture_D

class GestLate_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  audio/log_mel_512  = 32, 64, 128
  text/bert = 32, 64, 768
  '''
  def __init__(self, time_steps=64, in_channels = 768, out_feats=96, p=0, **kwargs):
    super().__init__()
    torch.set_default_tensor_type(torch.DoubleTensor)
    self.fc1 = nn.Sequential(nn.Linear(128 * 22 + 772, 612), nn.Dropout(0.2))
    self.fc2 = nn.Sequential(nn.Linear(612, 256),  nn.Dropout(0.2))
    self.fc3 = nn.Sequential(nn.Linear(256, 45),  nn.Dropout(0.2))
    self.fc_pose = nn.Sequential(nn.Linear(out_feats*3, 512), nn.Dropout(p=0.8))
    self.film_fc_a = nn.Linear(512, 45)
    self.film_fc_b = nn.Linear(512, 45)
    self.fc_fin = nn.Sequential(nn.Linear(45, out_feats), nn.Dropout(0.2), nn.Tanh())
    #speaker files

    with open('all_filler_avg.p', 'rb') as fp:
        filler_avg = pickle.load(fp)
    self.filler_avg = filler_avg



  def forward(self, x, y, time_steps=64, **kwargs):
    #Concatenate speech encodings into long vector
    #concatenate text_features to the long vector
    #fc 612, 256, 45 (w/o PCA)
    #3 previoius poses encodes into 512 dim conditional vector
    #activation function tanh,
    #dropout 0.2 each layer
    #pose encoding 0.8 dropout



    for i, modality in enumerate(kwargs['input_modalities']):
        if modality == "text/bert":
            t_i = i
        if modality.split('/')[0] == 'audio' and  modality != "audio/silence":
            a_i = i

    #implement silence masking of -15, if 1, then apply mask
    #implement filer masking of -15, if 1, apply mask
    if kwargs["sample_flag"] == 0:
        silence_mask = kwargs['audio/silence'].unsqueeze(-1)
        x[a_i] = x[a_i]*(1-silence_mask) - 15*silence_mask

        speaker_filler_avg = self.filler_avg[kwargs['speaker'][0]]
        filler_mask = kwargs['text/filler'].unsqueeze(-1)
        x[t_i] = x[t_i]*(1-filler_mask) + speaker_filler_avg.to(y.device)*filler_mask
    if kwargs["sample_flag"] == 1:
        silence_mask = kwargs['audio/silence'].reshape(1,-1,1)
        x[a_i] = x[a_i]*(1-silence_mask) - 15*silence_mask
        speaker_filler_avg = self.filler_avg[kwargs['speaker'][0]]
        filler_mask = kwargs['text/filler'].reshape(1,-1,1)
        x[t_i] = x[t_i]*(1-filler_mask) + speaker_filler_avg.to(y.device)*filler_mask






    #additional text features for gesticulator
    x[t_i] = F.pad(x[t_i], (0,4)) #introduce padding for 4 more features in the 3rd dimenssion
    if kwargs["sample_flag"] == 0:  #train/test loop
        text_duration_long = kwargs['text/token_duration'].long()
        for i, interval in enumerate(text_duration_long):
            new_pos = 0
            for j, word_dur in enumerate(interval):
                if word_dur.item() != 0:
                    fr_passed = torch.arange(1,word_dur+1)
                    x[t_i][i, new_pos: new_pos + word_dur,768] = fr_passed # timesteps from beginning of the word
                    x[t_i][i, new_pos: new_pos + word_dur,769] = fr_passed.float()/word_dur #relative progress of the word
                    x[t_i][i, new_pos: new_pos + word_dur,770] = word_dur - fr_passed #timesteps to the end of the word
                    x[t_i][i, new_pos: new_pos + word_dur,771] = word_dur # #duration of the word
                    new_pos += word_dur
    if kwargs["sample_flag"] == 1: #sample loop
        text_duration_long = kwargs['text/token_duration'].flatten()
        for word_dur in text_duration_long:
            new_pos = 0
            word_dur_int = int(word_dur)
            if word_dur_int != 0:
                fr_passed = torch.arange(1,word_dur_int+1)
                x[t_i][0, new_pos: new_pos + word_dur_int,768] = fr_passed # timesteps from beginning of the word
                x[t_i][0, new_pos: new_pos + word_dur_int,769] = fr_passed.float()/word_dur #relative progress of the word
                x[t_i][0, new_pos: new_pos + word_dur_int,770] = word_dur - fr_passed #timesteps to the end of the word
                x[t_i][0, new_pos: new_pos + word_dur_int,771] = word_dur # #duration of the word
                new_pos += word_dur_int



    new_x = torch.zeros(x[a_i].shape[0], x[a_i].shape[1], x[a_i].shape[2]*22).to(y.device)
    new_x_text = torch.zeros(x[a_i].shape[0], x[a_i].shape[1], x[a_i].shape[2]*22 +772).to(y.device)
    encoded_x = torch.zeros(x[a_i].shape[0], x[a_i].shape[1], 612).to(y.device)


    time_steps = x[a_i].shape[1]

    for i in range(0, time_steps): # sliding window implementation
        if i - 7 <= 0 : #early pad
            x_i = x[a_i][:,0:i+15,:].to(y.device)
            x_i = x_i.flatten(1)
            pad = torch.zeros(x[a_i].shape[0], abs(i-7), x[a_i].shape[2]).to(y.device)
            pad = pad.flatten(1)
            new_x[:,i,:] = torch.cat((pad, x_i), dim = 1)
        if i + 15 >= time_steps: #late pad
            x_i = x[a_i][:,i-7:,:].to(y.device)
            x_i = x_i.flatten(1)
            pad = torch.zeros(x[a_i].shape[0], i+15-time_steps, x[a_i].shape[2]).to(y.device)
            pad = pad.flatten(1)
            new_x[:,i,:] = torch.cat((x_i, pad), dim = 1)
        if i in range(7, time_steps-15):
            new_x[:,i,:] = x[a_i][:, i-7: i + 15, :].flatten(1)
        new_x_text[:,i,:] = torch.cat((new_x[:,i,:], x[t_i][:,i,:].to(x[t_i].device)), dim = 1) #add text for each iteration
        encoded_x[:,i,:] = self.fc1(new_x_text[:,i,:].clone())
    x = self.fc2(encoded_x)
    x = self.fc3(x)


    prev_pad = torch.zeros(y.shape[0], 1, y.shape[2]).to(y.device)
    torch.autograd.set_detect_anomaly(True)

    #training regiment
    if self.training == True:
        epoch = kwargs["epoch"]
        prev_0 = torch.cat([prev_pad]*3, 2).to(y.device).flatten(1)
        prev_1 = torch.cat([prev_pad]*2, 2).to(y.device).flatten(1)
        prev_2 = torch.cat([prev_pad]*1, 2).to(y.device).flatten(1)
        prev = torch.zeros_like(prev_0).to(y.device)
        fin_x = torch.zeros_like(y)
        if epoch in range(0, 6):
            for i in range(0,time_steps):
                fin_x[:,i,:] = self.fc_fin(x[:,i,:].clone())
        if epoch in range(6, 11): #changing teacher forcing amount per epoch (varies from 6~10 epochs)
            res = y.clone()
            if epoch == 6: # full teacher forcing
                for i in range(0,time_steps):
                    if i == 0:
                        prev = self.fc_pose(prev_0) # apply fully connected for prevous poses, set to 0 if not existent
                    if i == 1:
                        prev = self.fc_pose(torch.cat([prev_1, res[:,i-1:i,:].clone().flatten(1)], 1))
                    if i == 2:
                        prev = self.fc_pose(torch.cat([prev_2, res[:,i-2:i,:].clone().flatten(1)], 1))
                    if i > 2:
                        prev = self.fc_pose(res[:,i-3:i,:].clone().flatten(1))
                    a = self.film_fc_a(prev.unsqueeze(1))
                    b = self.film_fc_b(prev.unsqueeze(1))
                    a = a.squeeze(1)
                    b = b.squeeze(1)
                    x[:,i,:] = a*x[:,i,:].clone() + b
                    fin_x[:,i,:] = self.fc_fin(x[:,i,:].clone())
            else:
                res_update = 2**(4 -(epoch - 7)) # 16, 8, 4, 2
                for i in range(0,time_steps):
                    if i == 0:
                        prev = self.fc_pose(prev_0) # apply fully connected for prevous poses, set to 0 if not existent
                    if i == 1:
                        prev = self.fc_pose(torch.cat([prev_1, res[:,i-1:i,:].clone().flatten(1)], 1))
                    if i == 2:
                        prev = self.fc_pose(torch.cat([prev_2, res[:,i-2:i,:].clone().flatten(1)], 1))
                    if i > 2:
                        prev = self.fc_pose(res[:,i-3:i,:].clone().flatten(1))
                    a = self.film_fc_a(prev.unsqueeze(1))
                    b = self.film_fc_b(prev.unsqueeze(1))
                    a = a.squeeze(1)
                    b = b.squeeze(1)
                    x[:,i,:] = a*x[:,i,:].clone() + b
                    fin_x[:,i,:] = self.fc_fin(x[:,i,:].clone())
                    if (i % res_update == 0):
                        res[:,i,:] = fin_x[:,i,:].clone()
        else: # from 11th epoch, no teacher forcing, feeds in generated
            res = torch.zeros_like(y)
            for i in range(0,time_steps):
                if i == 0:
                    prev = self.fc_pose(prev_0)
                if i == 1:
                    prev = self.fc_pose(torch.cat([prev_1, res[:,i-1:i,:].clone().flatten(1)], 1))
                if i == 2:
                    prev = self.fc_pose(torch.cat([prev_2, res[:,i-2:i,:].clone().flatten(1)], 1))
                if i > 2:
                    prev = self.fc_pose(res[:,i-3:i,:].clone().flatten(1))
                a = self.film_fc_a(prev.unsqueeze(1))
                b = self.film_fc_b(prev.unsqueeze(1))
                a = a.squeeze(1)
                b = b.squeeze(1)
                x[:,i,:] = a*x[:,i,:].clone() + b
                fin_x[:,i,:] = self.fc_fin(x[:,i,:].clone())
                res[:,i,:] = fin_x[:,i,:].clone()

    #test (fully autoregressive) - same as 11th epoch onwards
    if self.training == False:
        prev_0 = torch.cat([prev_pad]*3, 2).to(y.device).flatten(1)
        prev_1 = torch.cat([prev_pad]*2, 2).to(y.device).flatten(1)
        prev_2 = torch.cat([prev_pad]*1, 2).to(y.device).flatten(1)
        prev = torch.zeros_like(prev_0).to(y.device)

        fin_x = torch.zeros_like(y)
        res = torch.zeros_like(y)
        for i in range(0,time_steps):
            if i == 0:
                prev = self.fc_pose(prev_0) # apply fully connected for prevous poses, set to 0 if not existent
            if i == 1:
                prev = self.fc_pose(torch.cat([prev_1, res[:,i-1:i,:].clone().flatten(1)], 1))
            if i == 2:
                prev = self.fc_pose(torch.cat([prev_2, res[:,i-2:i,:].clone().flatten(1)], 1))
            if i > 2:
                prev = self.fc_pose(res[:,i-3:i,:].clone().flatten(1))
            a = self.film_fc_a(prev.unsqueeze(1))
            b = self.film_fc_b(prev.unsqueeze(1))
            a = a.squeeze(1)
            b = b.squeeze(1)
            x[:,i,:] = a*x[:,i,:].clone() + b
            fin_x[:,i,:] = self.fc_fin(x[:,i,:].clone())
            res[:,i,:] = fin_x[:,i,:].clone()
    internal_losses = []
    return fin_x, internal_losses
