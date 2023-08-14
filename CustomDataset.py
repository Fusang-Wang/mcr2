from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg, download_file_from_google_drive, extract_archive
from torchvision.datasets.vision import VisionDataset
import numpy as np

import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union

import PIL
import torch
from tqdm import tqdm

# from .utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
# from .vision import VisionDataset


class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"
        print(self._images_folder)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()
        # print(image_ids)

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        # print(labels)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))
        # print(image_id_to_label)

        self._labels = []
        self._image_files = []
        self.data = []

        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            image_file = self._images_folder / f"image_{image_id:05d}.jpg"
            self._image_files.append(image_file)
            image = PIL.Image.open(image_file).convert("RGB")
            image = np.array(image.resize((32,32)))
            # print(image.shape)
            self.data.append(image)

        # print(self.data[0].shape)
        self.targets = self._labels
        self.data = np.array(self.data)
        # print(self.data.shape)
        # print("target", self.targets.shape)

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image, label = self.data[idx], self.targets[idx]
        image = PIL.Image.fromarray(image)
        # image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)

CSV = namedtuple("CSV", ["header", "index", "data"])

class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "   ", ("train", "valid", "test", "all"))]
        # print("using data split:",split_)
        splits = self._load_csv("list_eval_partition.txt")
        # print(splits.data.shape)
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask1 = slice(None) if split_ is None else (splits.data == split_).squeeze() # mask for train valid and test data
        # print("number of mask1",sum(mask1))
        # if mask == slice(None):  # if split == "all"
        #     self.filename = splits.index
        # else:
        #     self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

        # load from external files
        # mask = np.load(os.path.join(self.root, self.base_folder, "mask_attrs_0.npy"))
        # mask = np.array(mask, dtype=bool)
        # mask = torch.from_numpy(mask)
        # self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

        self.identity = identity.data[:]
        self.bbox = bbox.data[:]
        self.landmarks_align = landmarks_align.data[:]
        self.attr = attr.data[:]
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor") # map from {-1, 1} to {0, 1}
        self.attr_names = attr.header

        classes = np.array([8, 15, 31])
        attr = self.attr.cpu().detach().numpy()
        attr = attr[:, classes]
        self.num_attrs = int(len(classes))
        C = np.array([2 ** x for x in range(self.num_attrs)]).reshape(self.num_attrs,1)
        class_list = np.dot(attr, C)
        # print("class_list", class_list.shape)
        targets_ = np.squeeze(class_list).tolist()

        # generate mask for label class balance
        selected_labels = [0,1,2,3,4,5,6,7]
        num_imgs_per_class = 10000
        selected_pos = np.array([])
        mask2 = np.zeros(splits.data.shape[0],dtype=bool) # mask of label classes
        print(f'selected label:{selected_labels}')
        print(f'maxnumber of imgs per class:{num_imgs_per_class}')

        print("##################################################")
        print("statistic information for the whole celebA dataset")
        for i in selected_labels:
            temp = class_list == i
            print(f"class {i}: number {sum(temp)} before masking")
            pos_temp,_ = np.where(class_list == i)
            pos_temp = np.array(pos_temp)
            np.random.shuffle(pos_temp)
            selected_pos = np.concatenate((selected_pos, pos_temp[:num_imgs_per_class]))
        print("##################################################")

        selected_pos = np.array(selected_pos,dtype=int)
        mask2[selected_pos] = True
        mask2 = torch.from_numpy(mask2)

        mask = mask1 * mask2
        self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.targets = [targets_[i] for i in torch.squeeze(torch.nonzero(mask))]

        print("##################################################")
        print("statistic information for the CUNSTOM dataset")
        # show class information
        print("showing class information")
        attr_names = np.array(self.attr_names)[classes]
        print(attr_names)

        class_list = np.array(self.targets, dtype=int)
        for i in range(2**(len(classes))):
            temp = class_list == i
            # pos_temp,_ = np.where(class_list == i)
            # pos_temp = np.array(pos_temp)
            print(f"class {i}: number {sum(temp)}")
        print("##################################################")


        # generate data
        self.data: Any = []
        filename_bar = tqdm(self.filename, "loading images from data")
        for name in filename_bar:
            X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", name))
           
            # center crop with PIL
            width, height = X.size   # Get dimensions
            new_width, new_height = 158, 158
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            X = X.crop((left, top, right, bottom))

            X = X.resize((32, 32))
            X = np.array(X)
            self.data.append(X)
        self.data = np.array(self.data)
        print("datashape", len(targets_), self.data.shape, len(self.targets))



    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
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

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        extract_archive(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, target = self.data[index], self.targets[index]
        # print("label", label.shape)
        X = PIL.Image.fromarray(X)

        # target: Any = []
        # for t in self.target_type:
        #     if t == "label":
        #         target = label
        #     if t == "attr":
        #         # target.append(self.attr[index, :])
        #     elif t == "identity":
        #         target.append(self.identity[index, 0])
        #     elif t == "bbox":
        #         target.append(self.bbox[index, :])
        #     elif t == "landmarks":
        #         target.append(self.landmarks_align[index, :])
        #     else:
        #         # TODO: refactor with utils.verify_str_arg
        #         raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)
        #
        # if target:
        #     target = tuple(target) if len(target) > 1 else target[0]
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        # else:
        #     target = None
        # print(target)

        return X, target


    def __len__(self) -> int:
        return len(self.filename) # lens of the max index of the images

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)
