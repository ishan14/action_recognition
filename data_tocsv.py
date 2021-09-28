import os 
import pandas as pd
from sklean.utils import shuffle

#### preprocessing inputs from UCF-11 action dataset
def make_csv(path: str):
  data = []
  labels = os.listdir(path)
  encoding = 0
  c=0

  for el in os.listdir(path):
    label_folder = os.path.join(path, el)
    if os.path.isdir(label_folder):
      video_folders = os.listdir(label_folder)
      for video_folder in video_folders:
        if video_folder != 'Annotation':
          videos = os.listdir(os.path.join(label_folder, video_folder))
          for video in videos:
            video_path = os.path.join(label_folder, video_folder, video)
            extension = os.path.splitext(video_path)[1]
            if extension == '.mpg':
              video_reader = imageio.get_reader(video_path, 'ffmpeg')
              if len(video_reader) >=64:
                data.append([video_path, encoding])
                # writer.writerow([video_path, encoding])
              else:
                print(video_path, 'false')
      encoding += 1

  data_csv = pd.DataFrame(data, columns = ['video_path', 'label'])
  return data_csv

def split_train_test(csv_file) -> str:
  data = pd.read_csv(csv_file)
  count = len(data)
  
  data = shuffle(data)

  train_data = int(0.8 * count)
  test_data = int(0.2 * count)

  train_sample = data.iloc[:train_data, :]
  test_sample = data.iloc[:train_data + test_data, :]

  train_sample.to_csv('train_sample.csv', index = False)
  test_sample.to_csv('test_sample.csv', index = False)

  return 'Done parsing'
