import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
import time
from slowfast import *
from head import *

def train(resume_epoch: int, num_epoch: int, train_data: Tensor, test_data: Tensor, save_dir: str):

  lr = 0.001
  model = Model(11)
  criterion = nn.CrossEntropyLoss()   ##### define criterion (pose + rgb loss+ regularization)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
  
  epoch_loss_list = []
  epoch_accuracy_list = [] 

  if resume_epoch == 0:
    print("Training model from scratch.....")
  else:
    print("Training model from {} epoch".format(resume_epoch))
    checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                            map_location= lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])

  model.to(device)
  criterion.to(device)

  for epoch in range(resume_epoch, num_epoch):

    start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    scheduler.step()
    model.train()

    for (rgb, pose, label) in tqdm(train_data):
      pose = pose.to(device)
      rgb = rgb.to(device)
      label = label.to(device)

      optimizer.zero_grad()

      outputs = model(rgb, pose)

      probs = nn.Softmax(dim=1)(outputs)
      preds = torch.max(probs, 1)[1]
      loss = criterion(outputs, label)

      loss.backward()
      optimizer.step()

      running_loss += loss.item() * rgb.size(0)
      running_accuracy += torch.sum(preds == label)

      torch.save({'epoch': epoch+1,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict()},
                os.path.join(save_dir,'models', 'action_' + 'epoch-' + str(epoch) + '.pth.tar'))

    epoch_loss = running_loss / len(train_data)
    epoch_accuracy = running_accuracy.double() / len(train_data)

    print("Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, num_epoch, epoch_loss, epoch_accuracy))

    epoch_loss_list.append(epoch_loss)
    if epoch_loss == min(epoch_loss_list):
      torch.save({'epoch': epoch+1,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict()},
                os.path.join(save_dir,'models', 'action_' + 'epoch-' + str(epoch) + '.pth.tar'))
      print("Saving model: {}".format(os.path.join(save_dir,'models', 'action_' + 'epoch-' + str(epoch) + '.pth.tar')))
    
    model.eval()
    val_running_loss = 0.0
    val_running_accuracy = 0.0

    for (val_rgb, val_pose, val_label) in tqdm(test_data):
      val_rgb = val_rgb.to(device)
      val_pose = val_pose.to(device)
      val_label = val_label.to(device)

      with torch.no_grad():
        val_outputs = model(pose, rgb)
      val_probs = nn.Softmax(dim=1)(val_outputs)
      val_preds = torch.max(val_probs, 1)[1]
      val_loss = criterion(val_outputs, val_label)

      val_running_loss += val_loss.item() * val_rgb.size(0)
      val_running_correct += torch.sum(val_preds == val_label)

    val_epoch_loss = val_running_loss / len(test_data)
    val_epoch_accuracy = val_running_accuracy.double() / len(test_data)

    print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, num_epoch, val_epoch_loss, val_epoch_accuracy))

    end = time.time()
    print("Time taken for epoch {}: {} sec.".format(epoch+1, (end-start)/60))


