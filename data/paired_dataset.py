import os.path
import torch
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import util.util as util
from torchvision import transforms


class PairedDataset(BaseDataset):
    """
    This dataset class can load paired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.dir_A = os.path.join(opt.dataroot, self.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, self.phase + 'B')  # create a path '/path/to/data/trainB'

        self.scenes = sorted([scene for scene in os.listdir(self.dir_A)
                              if os.path.isdir(os.path.join(self.dir_A, scene))])

        if self.phase == 'train':
            self.A_paths = sorted([os.listdir(os.path.join(self.dir_A, scene)) for scene in self.scenes]
                                  [:min(opt.max_dataset_size, len(self.scenes))])
            self.B_paths = sorted([os.listdir(os.path.join(self.dir_B, scene)) for scene in self.scenes]
                                  [:min(opt.max_dataset_size, len(self.scenes))])
            self.size = len(self.scenes)  # size of dataset is number of scenes in training
        else:
            self.A_paths, self.B_paths = [], []
            for scene in self.scenes:
                for f in os.listdir(os.path.join(self.dir_A, scene)):
                    self.A_paths.append(os.path.join(self.dir_A, scene, f))
                for f in os.listdir(os.path.join(self.dir_B, scene)):
                    self.B_paths.append(os.path.join(self.dir_B, scene, f))
            self.size = len(self.A_paths)  # size of dataset is #images in A in val and test

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            gt (tensor)      -- its groundtruth image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
            gt_paths (str)   -- image paths
        """
        if self.phase == 'train':
            A_scene_index = index
            if self.opt.serial_batches:   # make sure index is within then range
                B_scene_index = (self.size // 2 + index) % self.size
            else:   # randomize the index for domain B to avoid fixed pairs.
                B_scene_index = random.randint(0, self.size - 1)

            A_image_index = random.randint(0, len(self.A_paths[A_scene_index]) - 1)
            B_image_index = random.randint(0, len(self.B_paths[B_scene_index]) - 1)
            gt_image_index = 0

            A_path = os.path.join(self.dir_A, self.scenes[A_scene_index], self.A_paths[A_scene_index][A_image_index])
            B_path = os.path.join(self.dir_B, self.scenes[B_scene_index], self.B_paths[B_scene_index][B_image_index])
            gt_path = os.path.join(self.dir_B, self.scenes[A_scene_index], self.B_paths[A_scene_index][gt_image_index])
        else:
            A_path = self.A_paths[index]
            B_path = os.path.join(self.dir_B, os.path.basename(os.path.dirname(A_path)), os.path.basename(A_path)[:-9] + 'C-000.png')
            gt_path = B_path
            assert B_path in self.B_paths

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        # Apply image transformation
        # For paired images, we use the same randomness for random crop and random horizontal flip transforms
        i, j, _, _ = transforms.RandomCrop.get_params(
            torch.zeros(A_img.size[0], self.opt.load_size, self.opt.load_size),
            output_size=(self.opt.crop_size, self.opt.crop_size))
        flip = random.random() > 0.5
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt, params={'crop_pos': (i, j), 'flip': flip})
        A = transform(A_img)
        B = transform(B_img)
        gt = transform(gt_img)

        return {'A': A, 'B': B, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path, 'gt_paths': gt_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.size
