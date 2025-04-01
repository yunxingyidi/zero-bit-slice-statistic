import os

import numpy as np
import torch as t
import torch.utils.data
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image


def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )
    train_dataset = t.utils.data.Subset(dataset, indices=train_indices)
    val_dataset = t.utils.data.Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset


def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

def load_data(cfg):
    """ 仅加载用于推理的测试数据 """
    tv_normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    if cfg.dataset == 'imagenet':
        test_transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        # test_set = tv.datasets.ImageFolder(root=os.path.join(cfg.path, 'val'), transform=test_transform)
        test_set = ImageNetInferenceDataset(root=os.path.join(cfg.path, 'val'), transform=test_transform)
    elif cfg.dataset == 'cifar10':
        test_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        test_set = tv.datasets.CIFAR10(cfg.path, train=False, transform=test_transform, download=True)

    else:
        raise ValueError('load_test_data does not support dataset %s' % cfg.dataset)

    test_loader = t.utils.data.DataLoader(
        test_set, batch_size=cfg.batch_size, num_workers=cfg.workers, pin_memory=True)

    return test_loader


class ImageNetInferenceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if f.endswith((".jpg", ".JPEG", ".png"))]  # 只读取图片

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path  # 这里返回图片的路径，方便后续推理时获取文件名

