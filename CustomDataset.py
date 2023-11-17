import random
from pathlib import Path
import PIL.Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg, \
    download_file_from_google_drive, extract_archive
from torchvision.datasets.vision import VisionDataset
import numpy as np

import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union

import PIL
import torch
from tqdm import tqdm
import pickle


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
            image = np.array(image.resize((32, 32)))
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
    # TODO Attention, the output of the dataset contains all the images and is on CPU!!
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
            balance: bool = True,
            download: bool = False,
            max_imgNum: int = 10000,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.balance = balance
        self.max_imgNum = max_imgNum

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
        splits = self._load_csv("list_eval_partition.txt")
        split_mask = slice(None) if split_ is None else (splits.data == split_).squeeze()  # mask for train/valid/test
        attr_csv = self._load_csv("list_attr_celeba.txt", header=1)

        device = attr_csv.data.device
        type = attr_csv.data.type()
        self.attr = attr_csv.data[:]
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")  # map from {-1, 1} to {0, 1}
        self.attr_names = attr_csv.header

        wanted_labels = torch.tensor([8, 21, 36]).to(device)  # black_hair, mouth_slightly_opened ,wearing_lipsticks
        attr = self.attr
        attr = attr[:, wanted_labels]
        self.num_attrs = int(wanted_labels.size(0))

        # create the transform matrix for onehot2class
        c = torch.tensor([2 ** x for x in range(self.num_attrs)]).reshape(self.num_attrs, 1).to(device)
        class_labels = torch.matmul(attr, c)
        targets_ = np.squeeze(class_labels).tolist()

        # generate mask for label class balance
        selected_labels = [0, 1, 2, 3, 4, 5, 6, 7]
        num_imgs_per_class = attr.size(0)
        if self.balance:
            num_imgs_per_class = self.max_imgNum
        selected_pos = []
        attrs_mask = torch.zeros(splits.data.shape[0], dtype=bool)  # mask of label classes
        # print(f'selected label:{selected_labels} for attrs {self.attr_names}')
        print(f'max number of imgs per class:{num_imgs_per_class}')
        for i in selected_labels:
            temp = class_labels == i
            print(f"class {i}: number {sum(temp)} of the whole celebA dataset")
            pos_temp, _ = torch.where(class_labels == i)
            pos_temp = pos_temp[:num_imgs_per_class]
            # TODO implement shuffle for data selection
            selected_pos.append(pos_temp)
        print("---")
        selected_pos = torch.cat(selected_pos, dim=0)
        attrs_mask[selected_pos] = True

        mask = torch.logical_and(split_mask, attrs_mask)
        self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.targets = [targets_[i] for i in torch.squeeze(torch.nonzero(mask))]

        print("Check Statistic Info of the Dataset")
        # print(f"Wanted Labels: {self.attr_names}")

        class_list = np.array(self.targets, dtype=int)
        for i in range(2 ** self.num_attrs):
            temp = class_list == i
            print(f" Num Images for Class {i}: {sum(temp)}")
        print("---")

        # generate data
        self.data: Any = []
        filename_bar = tqdm(self.filename, "loading images from data")
        for name in filename_bar:
            X = PIL.Image.open(os.path.join(self.root, "img_align_celeba", name))
            # center crop with PIL
            width, height = X.size  # Get dimensions
            new_width, new_height = 158, 158
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            X = X.crop((left, top, right, bottom))

            X = X.resize((128, 128))
            X = np.array(X)
            self.data.append(X)
        self.data = np.array(self.data)
        # print("datashape", len(targets_), self.data.shape, len(self.targets))

    def _load_csv(
            self,
            filename: str,
            header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, "Anno", filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            _, ext = os.path.splitext(filename)
            if ext not in [".zip", ".7z"]:
                fpath = os.path.join(self.root, "Anno", filename)
            else:
                fpath = os.path.join(self.root, filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, "img_align_celeba"))

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        extract_archive(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, target = self.data[index], self.targets[index]
        x = PIL.Image.fromarray(x)

        if self.transform is not None:
            # TODO Attention! the transform function here is predefined in train_func and by defaut size of img is 32
            x = self.transform(x)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x, target

    def __len__(self) -> int:
        return len(self.filename)  # lens of the max index of the images

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)


class Compare(VisionDataset):
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

    # TODO Attention, the output of the dataset contains all the images and is on CPU!!
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.

    def __init__(
            self,
            root: str = None,
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            max_imgnum: int = 1000
    ):

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.max_imgnum = max_imgnum

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        pickle_file = os.path.join(root, "res.pkl")
        file_list, target = self._load_pkl(pickle_file, max_imgnum)

        self.targets = target
        self.file_list = file_list

        # generate data
        self.data: Any = []
        filename_bar = tqdm(self.file_list, "loading images from data")
        for name in filename_bar:
            X = PIL.Image.open(os.path.join(self.root, "img", name))
            # center crop with PIL
            width, height = X.size  # Get dimensions
            new_width, new_height = 158, 158
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2
            X = X.crop((left, top, right, bottom))

            X = X.resize((128, 128))
            X = np.array(X)
            self.data.append(X)
        self.data = np.array(self.data)

    def _load_pkl(
            self,
            pickle_file: str,
            maxnum: int
    ):
        file_list = []
        pred_labels = []
        with open(pickle_file, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            random.shuffle(data)
            num_class = data[0]["num_classes"]
            for item in tqdm(data[: maxnum]):
                file_list.append(item["img_path"])
                pred_class = 0
                pred_ = item["pred_label"].detach().cpu().numpy()
                for class_ in pred_:
                    pred_class += 2 ** class_
                pred_labels.append(pred_class)

        return file_list, pred_labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        x, target = self.data[index], self.targets[index]
        x = PIL.Image.fromarray(x)

        if self.transform is not None:
            # TODO Attention! the transform function here is predefined in train_func and by defaut size of img is 32
            x = self.transform(x)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x, target

    def __len__(self) -> int:
        return len(self.file_list)  # lens of the max index of the images

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)
