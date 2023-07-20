import os
import glob
import functools
import numpy as np

class ImageFile(object):
    def __init__(self, phase='train'):
        self.phase = phase
        self.rng = np.random.RandomState(0)

    def _get_valid_names(self, *dirs, shuffle=True):
        # Extract valid names
        name_sets = [self._get_name_set(d) for d in dirs]

        # Reduce
        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))
        if shuffle:
            self.rng.shuffle(valid_names)

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        return [os.path.join(data_dir, name + ext)
                for name in data_list]


class ImageFileTrain(ImageFile):
    def __init__(self,
                 alpha_dir="train_alpha",
                 fg_dir="train_fg",
                 bg_dir="train_bg",
                 alpha_ext=".jpg",
                 fg_ext=".jpg",
                 bg_ext=".jpg"):
        super(ImageFileTrain, self).__init__(phase="train")

        self.alpha_dir  = alpha_dir
        self.fg_dir     = fg_dir
        self.bg_dir     = bg_dir
        self.alpha_ext  = alpha_ext
        self.fg_ext     = fg_ext
        self.bg_ext     = bg_ext

        self.valid_fg_list = self._get_valid_names(self.fg_dir, self.alpha_dir)
        self.valid_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.bg_dir)]

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_fg_list)
        self.fg = self._list_abspath(self.fg_dir, self.fg_ext, self.valid_fg_list)
        self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_bg_list)

    def __len__(self):
        return len(self.alpha)


class ImageFileTest(ImageFile):
    def __init__(self,
                 alpha_dir="test_alpha",
                 merged_dir="test_merged",
                 trimap_dir="test_trimap",
                 alpha_ext=".png",
                 merged_ext=".png",
                 trimap_ext=".png"):
        super(ImageFileTest, self).__init__(phase="test")

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.trimap_dir = trimap_dir
        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.trimap_ext = trimap_ext

        self.valid_image_list = self._get_valid_names(self.alpha_dir, self.merged_dir, self.trimap_dir, shuffle=False)

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list)
        self.merged = self._list_abspath(self.merged_dir, self.merged_ext, self.valid_image_list)
        self.trimap = self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list)

    def __len__(self):
        return len(self.alpha)


dataset = {'AIM', 'PPM', 'AM2k_train','AM2k_val', 'RWP636','P3M_val_np','P3M_train','P3M_val_p'}
#            o_      _o or _5k    m_           m_           _input.jpg   p_         p_         p_ 

def get_dir_ext(dataset):
    if  dataset == 'AIM':
        image_dir = '/home/yezixuan/Combined_Dataset/Test_set/AIM-500/original'
        label_dir = '/home/yezixuan/Combined_Dataset/Test_set/AIM-500/mask'
        trimap_dir = '/home/yezixuan/Combined_Dataset/Test_set/AIM-500/trimap'

        merged_ext = '.jpg'
        alpha_ext = '.png'
        trimap_ext = '.png'

    elif  dataset == 'PPM':
        image_dir = '/home/yezixuan/Combined_Dataset/PPM-100/image_train'
        label_dir = '/home/yezixuan/Combined_Dataset/PPM-100/matte_train'
        trimap_dir = '/home/yezixuan/Combined_Dataset/PPM-100/trimap'

        merged_ext = '.jpg'
        alpha_ext = '.jpg'
        trimap_ext = '.jpg'
    elif  dataset =='AM2k_train':
        image_dir = "/data/yezixuan/Combined_Dataset/AM-2k/train/original"
        label_dir = "/data/yezixuan/Combined_Dataset/AM-2k/train/mask"
        trimap_dir = "/data/yezixuan/Combined_Dataset/AM-2k/train/trimap"

        merged_ext = '.jpg'
        alpha_ext = '.png'
        trimap_ext = '.png'
    
    elif  dataset =='AM2k_val':
        image_dir = "/data/yezixuan/Combined_Dataset/AM-2k/validation/original"
        label_dir = "/data/yezixuan/Combined_Dataset/AM-2k/validation/mask"
        trimap_dir = "/data/yezixuan/Combined_Dataset/AM-2k/validation/trimap"

        merged_ext = '.jpg'
        alpha_ext = '.png'
        trimap_ext = '.png'
    elif  dataset =='RWP636':
        image_dir = '/data/yezixuan/Combined_Dataset/RealWorldPortrait-636/image'
        label_dir = '/data/yezixuan/Combined_Dataset/RealWorldPortrait-636/alpha'
        trimap_dir = '/data/yezixuan/Combined_Dataset/RealWorldPortrait-636/trimap'
        merged_ext = '.jpg'
        alpha_ext = '.png'
        trimap_ext = '.png'
    elif  dataset == 'P3M_val_np': # non blurred
        image_dir = '/data/yezixuan/Combined_Dataset/P3M-10k/validation/P3M-500-NP/original_image'
        label_dir = '/data/yezixuan/Combined_Dataset/P3M-10k/validation/P3M-500-NP/mask'
        trimap_dir = '/data/yezixuan/Combined_Dataset/P3M-10k/validation/P3M-500-NP/trimap'
        merged_ext = '.jpg'
        alpha_ext = '.png'
        trimap_ext = '.png'
    elif  dataset == 'P3M_val_p': # blurred
        image_dir = '/data/yezixuan/Combined_Dataset/P3M-10k/validation/P3M-500-P/blurred_image'
        label_dir = '/data/yezixuan/Combined_Dataset/P3M-10k/validation/P3M-500-P/mask'
        trimap_dir = '/data/yezixuan/Combined_Dataset/P3M-10k/validation/P3M-500-P/trimap'
        merged_ext = '.jpg'
        alpha_ext = '.png'
        trimap_ext = '.png'   
    elif  dataset == 'P3M_train': # blurred
        image_dir = '/data/yezixuan/Combined_Dataset/P3M-10k/train/blurred_image'
        label_dir = '/data/yezixuan/Combined_Dataset/P3M-10k/train/mask'
        trimap_dir = '/data/yezixuan/Combined_Dataset/P3M-10k/train/trimap'
        merged_ext = '.jpg'
        alpha_ext = '.png'
        trimap_ext = '.png'
        
    return image_dir, label_dir, trimap_dir, merged_ext, alpha_ext, trimap_ext  
    
class MultiImageFile(object):
    def __init__(self):
        
        self.rng = np.random.RandomState(1)

    def _get_valid_names(self, *dirs, shuffle=True):
        # Extract valid names
        name_sets = [self._get_name_set(d) for d in dirs]

        # Reduce
        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))
        
        # ensure the order is the same for both training and validation
        if shuffle:
            valid_names.sort()
            self.rng.shuffle(valid_names)

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)
            name = os.path.splitext(name)[0]
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        return [os.path.join(data_dir, name + ext)
                for name in data_list]

class MultiImageFileDoubleSet(MultiImageFile):
    def __init__(self,  ratio = 0.9, dataset_name = ['AIM', 'PPM', 'AM2k_train','AM2k_val', 'RWP636','P3M_val_np']):

        super(MultiImageFileDoubleSet, self).__init__()

        self.alpha_train = []
        self.merged_train = []
        self.trimap_train = []
        self.alpha_val = []
        self.merged_val = []
        self.trimap_val = []
        
        for dataset_name_ in dataset_name:
            merged_dir, alpha_dir, trimap_dir, merged_ext, alpha_ext, trimap_ext = get_dir_ext(dataset_name_)
            valid_image_list = self._get_valid_names(alpha_dir, merged_dir, trimap_dir)
            
            alpha = self._list_abspath(alpha_dir, alpha_ext, valid_image_list)
            merged = self._list_abspath(merged_dir, merged_ext, valid_image_list)
            trimap = self._list_abspath(trimap_dir, trimap_ext, valid_image_list)
            
            alpha_train, alpha_val = self._split(alpha, ratio)
            merged_train, merged_val = self._split(merged, ratio)
            trimap_train, trimap_val = self._split(trimap, ratio)
            
            self.alpha_train.extend(alpha_train)
            self.merged_train.extend(merged_train)
            self.trimap_train.extend(trimap_train)
            self.alpha_val.extend(alpha_val)
            self.merged_val.extend(merged_val)
            self.trimap_val.extend(trimap_val)

    def _split(self, data_list, ratio):
        num = len(data_list)
        split = int(num * ratio)
        return data_list[:split], data_list[split:]
    
    
if __name__ == "__main__":

    test = MultiImageFileDoubleSet()
    print(0)
