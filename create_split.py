# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import csv
import os
from collections import namedtuple
import numpy as np
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg, download_file_from_google_drive, extract_archive
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

CSV = namedtuple("CSV", ["header", "index", "data"])

# %%
dataroot = '/data/fusang/fm/celeba_raw/Anno'
image_size = 32

# %%
def _load_csv(
    filename: str,
    header = None,
):
    with open(os.path.join(dataroot, filename)) as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

    if header is not None:
        headers = data[header]
        data = data[header + 1 :]
    else:
        headers = []

    indices = [row[0] for row in data]
    data = [row[1:] for row in data]
    data_int = [list(map(int, i)) for i in data]

    return CSV(headers, indices, torch.tensor(data_int))

# %%
split_map = {
    "train": 0,
    "valid": 1,
    "test": 2,
    "all": None,
}
split = 'train'
split_ = split_map[verify_str_arg(split.lower(), "   ", ("train", "valid", "test", "all"))]

splits = _load_csv("list_eval_partition.txt")

identity = _load_csv("identity_CelebA.txt")

attr = _load_csv("list_attr_celeba.txt", header=1)

# %%
def find_subset_mask(remove_label, splits, attr, split_name):
    split_map = {
    "train": 0,
    "valid": 1,
    "test": 2,
    "all": None,
    }
    split_name = split_name
    split_ = split_map[verify_str_arg(split_name.lower(), "   ", ("train", "valid", "test", "all"))]
    mask1 = slice(None) if split_ is None else (splits.data == split_).squeeze() # mask for train valid and test data
    mask1 = mask1.cpu().detach().numpy()
    # print(mask1.shape)

    attr_names = attr.header
    indices = attr.index
    attr_ = attr.data[:]
    attr_ = torch.div(attr_ + 1, 2, rounding_mode="floor") # map from {-1, 1} to {0, 1}
    attr_ = attr_.cpu().detach().numpy()
    # print("attr_", attr_.shape)

    classes = np.array([8, 31])
    attr_ = attr_[:, classes]
    num_attrs = int(len(classes))
    # print("attr_", attr_.shape)
    C = np.array([2 ** x for x in range(num_attrs)]).reshape(num_attrs, 1)
    # print("C", C.shape)
    class_list = np.matmul(attr_, C)
    # print("Class_list", class_list.shape)
    
    selected_labels = [0,1,2,3]
    selected_labels.remove(remove_label)
    num_imgs_per_class = 1000000000
    # selected_pos = np.array([])
    mask2 = np.zeros(splits.data.shape[0],dtype=bool) # mask of label classes
    # print("mask2", mask2.shape)
    # print(f'selected label:{selected_labels}', mask2.shape)
    print(f'maxnumber of imgs per class:{num_imgs_per_class}')

    print("WHOLE DATASET INFO:")
    for i in selected_labels:
        indexor = class_list == i
        indexor = np.array(indexor, dtype=bool)
        indexor = np.squeeze(indexor)
        print(f"class {i}: number {sum(indexor)} ")
        mask2 = np.logical_or(mask2, indexor)
        # print("indexor", indexor.shape)

    assert mask1.shape == mask2.shape
    mask = np.logical_and(mask1, mask2)


    used_attr_names = np.array(attr_names)[classes]
    print("CUSTOM DATASET INFO:")
    print("showing class information")
    print(used_attr_names)

    # print(class_list.shape, mask.shape)
    class_list_selected = np.array(class_list, dtype=int)[mask]
    for i in range(2**(len(classes))):
        temp = class_list_selected == i
        print(f"class {i}: number {sum(temp)}")
        
    mask_path = f'mask_npy/celeba_{classes[0]}_{classes[1]}_no{remove_label}_{split_name}.npy'
    np.save(os.path.join("/data/fusang/fm/mcr2", mask_path), mask)
    print(f'mask saving to {mask_path}')

# %%
splits = _load_csv("list_eval_partition.txt")
attr = _load_csv("list_attr_celeba.txt", header=1)
find_subset_mask(0, splits, attr, 'train')

# %%
import imageio
from img_utils.image_transform import NumpyResize, pil_loader

def select_subset_images(mask_cls, inputPath, outputPath, maxNumber):
    classes = np.array([8, 31])
    attr = _load_csv("list_attr_celeba.txt", header=1)
    mask_cls = np.load(mask_cls)
    mask_cls = torch.Tensor(mask_cls)
    imgList = [attr.index[i] for i in torch.squeeze(torch.nonzero(mask_cls))]
    attrList = np.array([attr.data[i].cpu().numpy() for i in torch.squeeze(torch.nonzero(mask_cls))])
    attrList = (attrList[:, classes] + 1)//2

    num_attrs = int(len(classes))
    C = np.array([2 ** x for x in range(num_attrs)]).reshape(num_attrs, 1)
    class_list = np.matmul(attrList, C)
    print("Class_list", class_list.shape)
    numImgs = len(imgList)
    print('Number of Images:', numImgs)
    print("CHECK CUSTOM DATASET INFO:")
    for i in range(2**num_attrs):
        indexor = class_list == i
        indexor = np.array(indexor, dtype=bool)
        indexor = np.squeeze(indexor)
        print(f"class {i}: number {sum(indexor)} ")

    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    for index, item in enumerate(tqdm(imgList[:maxNumber])):
        in_path = os.path.join(inputPath, item)
        img = np.array(pil_loader(in_path))
        out_path = os.path.join(outputPath, item)
        imageio.imwrite(out_path, img)
    print("Finished saving subdataset to", outputPath)

# # %%
mask = "/data/fusang/fm/mcr2/mask_npy/celeba_8_31_no0_train.npy"
input = "/data/fusang/fm/celeba_raw/img_align_celeba"
ouput = "/data/fusang/fm/celeba_raw/celeba_blackH_smile_no0_train"
select_subset_images(mask, input, ouput, 10000000)