from train.py import *
from dataloader import *
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import sys


resume_epoch = sys.argv[1]
final_epoch = sys.argv[2]
b_size = sys.argv[3]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device available: {}".format(device))

train_data = dataset('train_sample.csv')
train_loader = DataLoader(train_data, batch_size=b_size)
test_data = dataset('test_sample.csv')
test_loader = DataLoader(test_data, batch_size=1)

train(resume_epoch, final_epoch, train_loader, test_loader)