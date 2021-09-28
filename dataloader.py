import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models, utils, get_video_backend
from PIL import Image
import imageio
import pandas as pd

from pose_extract import *


class dataset(Dataset):
  def __init__(self, csv_file):
    self.data = pd.read_csv(csv_file, index_col = False)
    self.body = body_pose('body_pose_model.pth')
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    video_path = self.data.iloc[idx, 1]
    label = self.data.iloc[idx, 2]
    label = np.array(label)

    buffer = self.load_frames(video_path)
    pose_buffer = self.pose_extract(buffer)
    buffer = self.normalize(buffer)
    buffer = self.to_tensor(buffer)
    pose_buffer = self.to_tensor(pose_buffer)

    return torch.from_numpy(buffer), torch.from_numpy(pose_buffer), torch.from_numpy(label)

  def load_frames(self, video_path):
    video_reader = imageio.get_reader(video_path, 'ffmpeg')
    frames = np.empty((64, 224, 224, 3), np.dtype('float32'))
    
    for idx in range(64):
      image = Image.fromarray(video_reader.get_data(idx))
      image = image.resize((224, 224))
      image = np.array(image, dtype = np.float32)
      frames[idx] = image

    return frames

  def normalize(self, buffer):
    
    for i, frame in enumerate(buffer):
      frame -= np.array([[[90, 98.0, 102.0]]])
      buffer[i] = frame
    return buffer

  def to_tensor(self, buffer):
    return buffer.transpose((3,0,1,2))

  def pose_extract(self, buffer):
    pose_frames = np.empty((64, 224, 224, 18), np.dtype('float32'))
    for i, frame in enumerate(buffer):
      heatmap = self.body(frame)
      pose_frames[i] = heatmap
    return pose_frames

  def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad