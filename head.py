import torch
import torch.nn as nn
import torch.nn.functional as F
from slowfast impoort *

class dense_block(nn.Module):
  def __init__(self, in_features, out_features):
    super(dense_block, self).__init__()
    
    self.in_features = in_features
    self.out_features = out_features
    self.linear = nn.Linear(self.in_features, self.out_features)
    self.activation = nn.LeakyReLU(negative_slope=0.01)
    self.drop = nn.Dropout(p=0.5)

  def forward(self, x):
    x = self.linear(x)
    x = self.activation(x)
    x = self.drop(x)
    return x

class Model(nn.Module):
  def __init__(self, num_classes: int):
    super(Model, self).__init__()

    self.num_classes = num_classes
    self.dense1 = dense_block(2304*14*14, 1024)
    self.dense2 = dense_block(1024, 512)
    self.dense3 = dense_block(512, 256)
    self.dense4 = dense_block(256, 128)
    # self.dense5 = dense_block(128, self.num_classes)
    self.dense5 = nn.Linear(128, num_classes)
    
    self.sf = SlowFast()
    # self.pm = bodypose_model()
  
  def forward(self, rgb, pose): ## we can also take depth
    # pose_out = pm(pose)
    fast_feature, slow_feature = self.sf(pose, rgb)
    fast_feature = nn.AvgPool3d(kernel_size=(fast_feature.shape[2], 1, 1))(fast_feature).squeeze(2)
    slow_feature = nn.AvgPool3d(kernel_size=(slow_feature.shape[2], 1, 1))(slow_feature).squeeze(2)

    feature = torch.cat([fast_feature, slow_feature], dim=1)
    
    ft = feature.view(feature.size(0), -1)
    

    ft = self.dense1(ft)
    ft = self.dense2(ft)
    ft = self.dense3(ft)
    ft = self.dense4(ft)
    ft = self.dense5(ft)
    return ft

