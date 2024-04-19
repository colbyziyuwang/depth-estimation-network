#im_left = image.imread("dataset/image_L/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png")
#im_right = image.imread("dataset/image_R/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png")
#im_gt = image.imread("dataset/disparity/2018-07-11-14-48-52_2018-07-11-15-09-57-367.png")\

import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg

# assumption of the github dataset directories:
# - dataset
# - - image_L
# - - image_R
# - - disparity

DATASET_PATH = 'dataset'
LEFT = 'image_L'
RIGHT = 'image_R'
DISPARITY = 'disparity'

class CustomDataset(Dataset):
    def __init__(self, root_dir='dataset', sub_dirs=None, train=True, transform=None):
        if sub_dirs is None:
            sub_dirs = [LEFT, RIGHT, DISPARITY]
        self.data_paths = {sub_dir: [] for sub_dir in sub_dirs}
        self.isTrainMode = train
        self.transform = transform
        for sub_dir in sub_dirs:
            full_dir_path = os.path.join(root_dir, sub_dir)
            for imagename in sorted(os.listdir(full_dir_path)):
                image_path = os.path.join(full_dir_path, imagename)
                self.data_paths[sub_dir].append(image_path)
        
    def __len__(self):
        return len(self.data_paths[LEFT])

    def __getitem__(self, idx):
        sample = {LEFT: mpimg.imread(self.data_paths[LEFT][idx])}
        
        if self.isTrainMode:
            sample[RIGHT] = mpimg.imread(self.data_paths[RIGHT][idx])
            sample[DISPARITY] = mpimg.imread(self.data_paths[DISPARITY][idx])

        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Example use case
training_dataset = CustomDataset(root_dir=DATASET_PATH)
train_loader = DataLoader(training_dataset, batch_size=4, shuffle=True)  # Set a valid batch size

def train(train_loader, model, optimizer):
    for i, sample in enumerate(train_loader):
        left_img, right_img = sample[LEFT], sample[RIGHT]
        # Training loop logic here...
