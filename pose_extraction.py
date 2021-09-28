import numpy as np
from collections import OrderedDict
from torch import Tensor
import torch
import torch.nn as nn

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([
                      ('conv1_1', [3, 64, 3, 1, 1]),
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('conv3_4', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                      ('conv4_4_CPM', [256, 128, 3, 1, 1])
                  ])


        # Stage 1
        block1_1 = OrderedDict([
                        ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
                    ])

        block1_2 = OrderedDict([
                        ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
                    ])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                    ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
                ])

            blocks['block%d_2' % i] = OrderedDict([
                    ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']


    def forward(self, x):

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2


class body_pose():
  def __init__(self, model_path):
    self.model = bodypose_model()
    if torch.cuda.is_available():
      self.model = self.model.cuda()
    model_dict = self.transfer(torch.load(model_path))

    self.model.load_state_dict(model_dict)
    self.model.eval()

    self.stride = 8
    self.padValue = 128

  def __call__(self, image):
    # imageToTest_padded, pad = self.padRightDownCorner(image)#, stride, padValue)
    # im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
    im = np.transpose(np.float32(image[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
    im = np.ascontiguousarray(im)

    data = torch.from_numpy(im).float()
    if torch.cuda.is_available():
        data = data.cuda()
    # data = data.permute([2, 0, 1]).unsqueeze(0).float()
    with torch.no_grad():
        limbs, joints = self.model(data)
    joints = joints.cpu().numpy()
    heatmap = np.transpose(np.squeeze(joints), (1, 2, 0))
    heatmap = cv2.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    return heatmap[:, :, :-1]

  

  def transfer(self, model_weights):
    transfered_model_weights = {}
    for weights_name in self.model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

  def padRightDownCorner(self, img):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % self.stride == 0) else self.stride - (h % self.stride) # down
    pad[3] = 0 if (w % self.stride == 0) else self.stride - (w % self.stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + self.padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + self.padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + self.padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + self.padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad
