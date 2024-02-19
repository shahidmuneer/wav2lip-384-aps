#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train_syncnet.py
# Created Date: Saturday Dec 24 2023
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 27sth Jan 2024 10:46:46 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import dirname, join, basename, isfile
from models.model_transformer import SyncTransformer
from sklearn.metrics import f1_score
from glob import glob

import numpy as np
import soundfile as sf

import torch
import torch.multiprocessing
from   torch.backends import cudnn

from torch import nn
from torch import optim
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter
from torchaudio.transforms import MelScale

import random, cv2, argparse
from hparams import hparams, get_image_list, get_image_list_new
from natsort import natsorted

torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator on LRS2')
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset",
                    default="/mnt/sdb/cxh/liwen/DATA/lrs2_preprocessed")
parser.add_argument("--test_data_root", help="Root folder of the preprocessed LRS2 dataset",
                    default="/mnt/DATA/dataset/LRS2/preprocessed")
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint',
                    default=None,
                    type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory',
                    default='./experiments/transformer',
                    type=str)
parser.add_argument('-c', '--cuda', type=int, default=0) # >0 if it is set as -1, program will use CPU

args = parser.parse_args()
writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'tensorboard'))


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

num_audio_elements = 3200  # 6400  # 16000/25 * syncnet_T
tot_num_frames = 25  # buffer
v_context = 5  # 10  # 5
BATCH_SIZE = 128  # 128
MODE = 'train'
TOP_DB = -hparams.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))
melscale = MelScale(n_mels=hparams.num_mels, sample_rate=hparams.sample_rate, f_min=hparams.fmin, f_max=hparams.fmax,
                    n_stft=hparams.n_stft, norm='slaney', mel_scale='slaney').to(0)
logloss = nn.BCEWithLogitsLoss()


class Dataset(object):
    def __init__(self, data_root, split):
        self.split = split
        if split == 'pretrain':
            # self.all_videos = get_image_list(data_root,split)
            self.all_videos = get_image_list_new(split)
        else:
            # self.all_videos = get_image_list(data_root,split)
            self.all_videos = get_image_list_new(split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_wav(self, wavpath, vid_frame_id):
        aud = sf.SoundFile(wavpath)
        can_seek = aud.seekable()
        pos_aud_chunk_start = vid_frame_id * 640
        _ = aud.seek(pos_aud_chunk_start)
        wav_vec = aud.read(num_audio_elements)
        return wav_vec

    def rms(self, x):
        val = np.sqrt(np.mean(x ** 2))
        if val==0:
            val=1
        return val

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + v_context):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            wavpath = join(vidname, "audio.wav")
            img_names = natsorted(list(glob(join(vidname, '*.jpg'))), key=lambda y: y.lower())
            interval_st, interval_end = 0, len(img_names)
            if interval_end-interval_st <= tot_num_frames:
                continue
            pos_frame_id = random.randint(interval_st, interval_end-v_context)
            pos_wav = self.get_wav(wavpath, pos_frame_id)
            rms_pos_wav = self.rms(pos_wav)

            img_name = os.path.join(vidname, str(pos_frame_id)+'.jpg')
            window_fnames = self.get_window(img_name)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue
            if random.choice([True, False]):
                y = torch.ones(1).float()
                wav = pos_wav
            else:
                y = torch.zeros(1).float()
                try_counter = 0
                while True:
                    neg_frame_id = random.randint(interval_st, interval_end - v_context)
                    if neg_frame_id != pos_frame_id:
                        wav = self.get_wav(wavpath, neg_frame_id)
                        if rms_pos_wav > 0.01:
                            break
                        else:
                            if self.rms(wav) > 0.01 or try_counter>10:
                                break
                        try_counter += 1

                if try_counter > 10:
                    continue
            aud_tensor = torch.FloatTensor(wav)

            # H, W, T, 3 --> T*3
            vid = np.concatenate(window, axis=2) / 255.
            vid = vid.transpose(2, 0, 1)
            vid = torch.FloatTensor(vid[:, 48:])
            if torch.any(torch.isnan(vid)) or torch.any(torch.isnan(aud_tensor)):
                continue
            if vid==None or aud_tensor==None:
                continue
            # print("vid shape:", vid.shape)
            # print("aud_tensor shape:", aud_tensor.shape)
            print(" ")
            # print("y shape:", y.shape)
            return vid, aud_tensor, y


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    import time
    import datetime    
    start_time  = time.time()

    # Caculate the epoch number
    print("Total epoch = %d"%nepochs)

    print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    while global_epoch < nepochs:
        f1_scores = []
        running_loss = 0.
        for step, (vid, aud, y) in enumerate(train_data_loader):
            vid = vid.to(device)
            gt_aud = aud.to(device)

            spec = torch.stft(gt_aud, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size,
                              window=torch.hann_window(hparams.win_size).to(gt_aud.device), return_complex=True)
            melspec = melscale(torch.abs(spec.detach().clone()).float())
            melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
            normalized_mel = torch.clip((2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
                                        -hparams.max_abs_value, hparams.max_abs_value)
            # print(normalized_mel.shape)
            mels = normalized_mel[:, :, :-1].unsqueeze(1)
            # print(mels.shape)
            model.train()
            optimizer.zero_grad()
            # print(vid.shape,mels.shape)
            out = model(vid.clone().detach(), mels.clone().detach())
            loss = logloss(out, y.squeeze(-1).to(device))
            loss.backward()
            optimizer.step()

            est_label = (out > 0.5).float()
            f1_metric = f1_score(y.clone().detach().cpu().numpy(),
                                 est_label.clone().detach().cpu().numpy(),
                                 average="weighted")
            f1_scores.append(f1_metric.item())
            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed)) 
        print('Elapsed [{}], Epoch:{}, [TRAINING LOSS]: {}, [TRAINING F1]: {}'
                                .format(elapsed, global_epoch, running_loss / (step + 1), sum(f1_scores)/len(f1_scores)))
        f1_epoch = sum(f1_scores) / len(f1_scores)
        writer.add_scalars('f1_epoch', {'train': f1_epoch}, global_epoch)
        writer.add_scalars('loss_epoch', {'train': running_loss / (step + 1)}, global_epoch)

        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

        with torch.no_grad():
            eval_model(test_data_loader, device, model, checkpoint_dir)

        global_epoch += 1


def eval_model(test_data_loader, device, model, checkpoint_dir, nepochs=None):
    losses = []
    running_loss=0
    f1_scores = []
    for step, (vid, aud, y) in enumerate(test_data_loader):
        model.eval()
        with torch.no_grad():
            vid = vid.to(device)
            gt_aud = aud.to(device)

            spec = torch.stft(gt_aud, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size,
                              window=torch.hann_window(hparams.win_size).to(gt_aud.device), return_complex=True)
            melspec = melscale(torch.abs(spec.detach().clone()).float())
            melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
            normalized_mel = torch.clip((2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
                                        -hparams.max_abs_value, hparams.max_abs_value)
            mels = normalized_mel[:, :, :-1].unsqueeze(1)
            out = model(vid.clone().detach(), mels.clone().detach())
            loss = logloss(out, y.squeeze(-1).to(device))
            losses.append(loss.item())

            est_label = (out > 0.5).float()
            f1_metric = f1_score(y.clone().detach().cpu().numpy(),
                                 est_label.clone().detach().cpu().numpy(),
                                 average="weighted")
            f1_scores.append(f1_metric.item())
            running_loss += loss.item()
    print('[VAL RUNNING LOSS]: {}, [VAL F1]: {}'
                                .format(running_loss / (step + 1), sum(f1_scores)/len(f1_scores)))
    averaged_loss = sum(losses) / len(losses)
    writer.add_scalars('loss_epoch', {'val': averaged_loss}, global_epoch)
    writer.add_scalars('f1_epoch', {'val': sum(f1_scores) / len(f1_scores)}, global_epoch)
    return


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    cudnn.benchmark = True
    cudnn.enabled   = True
    
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
    # Dataset and Dataloader setup
    train_dataset = Dataset(args.data_root,'train')
    test_dataset = Dataset(args.data_root,'val')
    # train_dataset = Dataset(args.data_root,"train_HDTF")
    # test_dataset = Dataset(args.data_root,"evaluate_HDTF")
    # train_dataset = Dataset(args.data_root,"train_avspeech")
    # test_dataset = Dataset(args.data_root,"evaluate_avspeech")

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=12)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=12)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncTransformer().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=5e-5)                       

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=100,
          nepochs=1000)