import os.path
from torch.utils.data import Dataset
import nibabel as nib
from multiprocessing import Manager
import random
import torchio as tio
import pickle
import torch

error_list = [5298, 5894,  393,   66,   68, 6576,   76, 6653,  347, 6218,  804,  844,
        5445,  751, 6075, 5501,  368,  236,  269, 5472,  494, 6711, 5571, 6543,
        5837, 5586, 6663,  791, 6113,  318,  299,  688,  323,   26, 6496, 5746]
def Generate_dataset():

    # This if else sentence is used to decide whether enables dataset cache funciton
    if False:
        cache = DatasetCache(None,use_cache=False)
        manager2 = Manager()
        cache2 = DatasetCache(manager2,use_cache=True)
    else:
        manager = None
        cache = DatasetCache(manager,use_cache=False)
        manager2 = None
        cache2 = DatasetCache(manager2,use_cache=False)

    # Loading all availible files
    if os.path.exists('dataset/pretrain_files_list.pkl'):
        with open('dataset/pretrain_files_list.pkl', 'rb') as file:
            files_list = pickle.load(file)

        # splitting the training and testing datasets. The Oasis will be treated as testing while the rest of them will be trated as training dataset.
        testing_set = files_list[0:1]

        training_sets = files_list
        del training_sets[0]

        training_sets[2] = [i.replace('GSP','GSP/FS_4p5') for i in training_sets[2]]
        training_sets_path = sum(training_sets, [])
        for index in sorted(error_list, reverse=True):
            del training_sets_path[index]
        training_sets_path= ['/scratch/datasets/xh278/orig'+i for i in training_sets_path]
        testing_set_path = sum(testing_set, [])

        train_dataset = VoxelDataset(training_sets_path, cache=cache, train=True)
        val_dataset = VoxelDataset(testing_set_path, cache=cache2, train=False)

    return train_dataset,val_dataset

class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        if self.manager is not None:
            self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, img, lbl):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (img, lbl)


class VoxelDataset(Dataset):
    def __init__(self, norm_file_path,  rescale_sdf=True, cache = None,train=True):
        # note that input image paths are for processed images rather than unprocessed
        self.norm_file_path = [i.replace('proc/ForAdrian_Talairach','orig').replace('talairach/','') for i in norm_file_path]
        self.orig_file_path = [i.replace('norm.mgz','orig.mgz').replace('talairach/','') for i in self.norm_file_path]

        self.intensity_spatial_norm_file_path = [i.replace('orig.mgz','talairach/norm.mgz') for i in self.orig_file_path]
        self.intensity_norm_file_path =[i.replace('orig.mgz','norm.mgz') for i in self.orig_file_path]
        self.seg_path =[i.replace('orig.mgz','talairach/aseg.mgz') for i in self.orig_file_path]

        removal = []
        for i in range(len(self.seg_path)):
            if not os.path.exists(self.seg_path[i]):
                removal.append(i)
        removal = []

        for i in reversed(removal):
            self.seg_path.pop(i)
            self.intensity_spatial_norm_file_path.pop(i)
            self.orig_file_path.pop(i)
        self.rescale_sdf = rescale_sdf
        self.cache = cache
        self.train = train
        self.transform = tio.Compose([
            tio.RandomGamma(log_gamma = (-0.3,0.3),p=0.3),
            ]
        )
    def __len__(self):
        return len(self.orig_file_path)

    def __getitem__(self, index):
        image_resolution= 256
        normalization=255
        output_index = [index]
        for cur_index in output_index:
            array = nib.load(self.orig_file_path[cur_index])
            input = array.get_fdata()
            input = torch.FloatTensor(input)
            input = input.unsqueeze(0).unsqueeze(0)
            if image_resolution != 256:
                input = torch.nn.functional.interpolate(input, size=[image_resolution, image_resolution, image_resolution], mode='trilinear',align_corners=False)[0]
            else:
                input = torch.Tensor(input)[0]

            array = nib.load(self.intensity_spatial_norm_file_path[cur_index])
            intensity_spatial_norm = array.get_fdata()
            intensity_spatial_norm = torch.FloatTensor(intensity_spatial_norm)
            intensity_spatial_norm = intensity_spatial_norm.unsqueeze(0).unsqueeze(0)
            if image_resolution != 256:
                intensity_spatial_norm = torch.nn.functional.interpolate(intensity_spatial_norm, size=[image_resolution, image_resolution, image_resolution], mode='trilinear',align_corners=False)[0]
            else:
                intensity_spatial_norm = torch.Tensor(intensity_spatial_norm)[0]

            seg = intensity_spatial_norm.unsqueeze(0)
            if image_resolution != 256:
                seg = torch.nn.functional.interpolate(seg, size=[image_resolution, image_resolution, image_resolution], mode='nearest')[0]
            else:
                seg = torch.Tensor(seg)[0]
        if self.train:
            subject = tio.Subject(image = tio.ScalarImage(tensor=input))
            transformed = self.transform(subject)
            input = transformed['image'].data
            input = input.clip(0, normalization) / normalization
            intensity_spatial_norm = intensity_spatial_norm.clip(0, normalization) / normalization
        else:
            input = input.clip(0, normalization) / normalization
            intensity_spatial_norm = intensity_spatial_norm.clip(0, normalization) / normalization
        return input,intensity_spatial_norm,seg,cur_index



