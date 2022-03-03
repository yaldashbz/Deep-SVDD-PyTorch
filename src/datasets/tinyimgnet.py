from PIL import Image
from torch.utils.data import Dataset

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import global_contrast_normalization

import os
import torch
import numpy as np
import torchvision.transforms as transforms

DATA_DIR = 'tiny-imagenet-200'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MAX_CLASSES = 20


class TinyImgNet_Dataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(28),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],
                                                             [min_max[normal_class][1] - min_max[normal_class][0]])])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = TinyImgNet(root=self.root, train=True, download=True,
                                    transform=transform, target_transform=target_transform)

        self.test_set = TinyImgNet(root=self.root, train=False, download=True,
                                   transform=transform, target_transform=target_transform)


class TinyImgNet(Dataset):
    base_folder = 'tiny-imagenet-200'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download and not self._check_exists():
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        wp = open(os.path.join(self.root, DATA_DIR, 'wnids.txt'), 'r')
        self.wnids = list(map(lambda w: w[:-1], wp.readlines()))

        # Create dictionary to store img filename (word 0) and corresponding
        # label (word 1) for every line in the txt file (as key value pair)
        val_path = os.path.join(self.root, VALID_DIR, 'images')
        train_path = os.path.join(self.root, TRAIN_DIR)

        fp = open(os.path.join(self.root, VALID_DIR, 'val_annotations.txt'), 'r')
        data = fp.readlines()
        self._preprocess_load_data(data, val_path)
        fp.close()

        if self.train:
            self.train_data, self.train_label = self._load_data(train_path, transform=transform, train=True)
        else:
            # use val instead of test
            self.test_data, self.test_label = self._load_data(val_path, transform=transform, train=False)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder, 'train')) and \
               os.path.exists(os.path.join(self.root, self.base_folder, 'test'))

    def _load_data(self, path, transform, train: bool = True, max_classes=20):
        data = []
        labels = []
        cls_directories = os.listdir(path)
        np.random.shuffle(cls_directories)
        cls_directories = cls_directories[:max_classes]

        for cls_dir in cls_directories:
            cls_path = os.path.join(path, cls_dir)
            if train:
                cls_path = os.path.join(cls_path, 'images')
            for img_file in os.listdir(cls_path):
                if img_file.endswith('.JPEG'):
                    img_path = os.path.join(cls_path, img_file)
                    img = Image.open(os.path.join(img_path))
                    if transforms.ToTensor()(img).size()[0] == 3:
                        img = transforms.Grayscale()(img)
                    img = transform(img)
                    img = img.reshape(28, 28)
                    data.append(img)
                    labels.append(self._class_to_idx(cls_dir))
        return torch.stack(data), torch.tensor(labels)

    def _class_to_idx(self, class_name):
        return self.wnids.index(class_name)

    @classmethod
    def _preprocess_load_data(cls, data, img_dir):
        img_dict = {}
        for line in data:
            words = line.split('\t')
            img_dict[words[0]] = words[1]

        for img, folder in img_dict.items():
            newpath = (os.path.join(img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(img_dir, img)):
                os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

    def download(self):
        import urllib
        import zipfile

        path = os.path.join(self.root, self.filename)
        urllib.request.urlretrieve(self.url, path)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
