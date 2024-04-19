#im_left = image.imread("dataset/image_L/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png")
#im_right = image.imread("dataset/image_R/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png")
#im_gt = image.imread("dataset/disparity/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png")\
import matplotlib.image as image
import os
from torch.utils.data import DataLoader

DATASET_PATH = 'dataset'
LEFT = 'image_L'
RIGHT = 'image_R'
DISPARITY = 'disparity'


# assumtion of the github dataset directories:
# - dataset
# - - image_L
# - - image_R
# - - disparity

class CustomDataset():
    def __init__(self, root_dir='dataset', sub_dirs = [LEFT, RIGHT, DISPARITY], train=True, transform=None):
        self.data_paths = {}
        self.isTrainMode = train
        self.transform = transform
        for sub_dir in os.listdir(root_dir):
            if sub_dir not in sub_dirs:
                continue

            for imagename in os.listdir(sub_dir):
                image_path = os.path.join(root_dir, sub_dir, imagename)
                self.data_paths[sub_dir].append(image_path)
        
    def __len__(self):
        return len(self.data_paths[LEFT])

    def __getitem__(self, idx):
        sample = {}

        sample[LEFT] = image.imread(self.data_paths[LEFT][idx])

        if self.isTrainMode:
            sample[RIGHT] = image.imread(self.data_paths[RIGHT][idx])

        if self.transform:
            sample = self.transform(sample)
        
        return sample

''' example use case '''
training_dataset = CustomDataset()
train_loader = DataLoader(training_dataset, batch_size=0, shuffle=True)

def train(train_loader, model, optimizer ...):
    for i, sample in enumerate(train_loader):
        left_img, right_img = sample[LEFT], sample[RIGHT]
