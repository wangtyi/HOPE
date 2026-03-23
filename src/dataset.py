import os
from PIL import Image
import zarr
import tifffile
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset


class _BaseLargeImageDataset(Dataset):
    """
    Base class providing shared image I/O for large-image datasets.
    """
    _cache = OrderedDict()
    _max_cache = 30
    _image_sizes = OrderedDict()

    def get_image_size(self, img_path):
        if img_path in self._image_sizes:
            width = self._image_sizes[img_path]["width"]
            height = self._image_sizes[img_path]["height"]
            return width, height
        
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(img_path)
            width, height = img.size
        elif img_path.lower().endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(img_path) as tif:
                page = tif.pages[0]
                height, width = page.shape[:2]
        else:
            img = zarr.open(img_path, mode="r")
            if img.shape[0] == 3:
                height, width = img.shape[1:]
            else:
                height, width = img.shape[:2]
        
        self._image_sizes[img_path] = {"width": width, "height": height}
        return width, height
    
    def get_image_object(self, img_path):
        if img_path in self._cache:
            return self._cache[img_path]["obj"]

        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            obj = Image.open(img_path).convert("RGB")

            if len(self._cache) >= self._max_cache:
                self._cache.popitem(last=False)
            self._cache[img_path] = {"obj": obj}
            return obj

        elif img_path.lower().endswith(('.tif', '.tiff')):
            tif = tifffile.TiffFile(img_path)
            page = tif.pages[0]
            store = page.aszarr()   # ZarrTiffStore
            z = zarr.open(store, mode="r")
            
            if len(self._cache) >= self._max_cache:
                old = self._cache.popitem(last=False)
                if "tif" in old[1]:
                    old[1]["tif"].close()
            self._cache[img_path] = {
                "obj": z,
                "tif": tif      # keep file handle alive
            }
            return z

        else:  # native zarr
            z = zarr.open(img_path, mode="r")
            if z.shape[0] == 3:    # CHW -> HWC
                z = np.transpose(z, (1, 2, 0))
            
            if len(self._cache) >= self._max_cache:
                self._cache.popitem(last=False)
            self._cache[img_path] = {"obj": z}
            return z
    

class PatchDataset(_BaseLargeImageDataset):
    """
    Dataset for loading patches with target and (optional) auxiliary labels.
    
    Supports PNG, TIFF, and zarr formats.

    Args:
        img_names: List of image file names or zarr folders.
        target_labels: List of target labels corresponding to images.
        aux_labels: List of auxiliary labels corresponding to images (optional, default=[]).
        transform: Transform to apply to images.
        dataset_dir: Directory containing `img_names`.
        is_crop: Whether to crop the images using `crop_range`.
        crop_range: List of crop coordinates [left, top, right, bottom] for each image.
        return_region: Whether to return crop coordinates.

    Returns:
        A dictionary containing the processed image and labels, optionally including crop coordinates.
    """
    def __init__(
        self, 
        img_names, 
        target_labels, 
        aux_labels=[], 
        transform=None,
        dataset_dir=None, 
        is_crop=False, 
        crop_range=None, 
        return_region=False
    ):
        self.img_names = img_names
        self.labels = target_labels
        self.aux_labels = aux_labels
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.is_crop = is_crop
        self.crop_range = crop_range
        self.return_region = return_region

        self.img_paths = []
        self.img_labels = []
        self.img_aux_labels = []

        assert len(self.labels) == len(self.img_names), 'The number of target labels and images is inconsistent.'
        if len(self.aux_labels) != 0:
            assert len(self.aux_labels) == len(self.img_names), 'The number of aux_labels and images is inconsistent.'

        self.process_with_labels()

    def process_with_labels(self):
        for idx, label in enumerate(self.labels):
            aux_label = self.aux_labels[idx] if len(self.aux_labels) > 0 else None
            img_name = self.img_names[idx]
            if str(label) != 'nan' and label is not None:
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    img_p = os.path.join(self.dataset_dir, img_name)
                else:
                    # zarr folder
                    img_p = os.path.join(self.dataset_dir, img_name, "he_img")
                if os.path.exists(img_p):
                    self.img_paths.append(img_p)
                    self.img_labels.append(label)
                    if aux_label is not None:
                        self.img_aux_labels.append(aux_label)
                else:
                    print(f"{img_p} doesn't exist and has been skipped.")
    
    def adjust_crop_box(self, box, width, height):
        left, top, right, bottom = box
        left = max(0, left)
        top = max(0, top)
        right = min(right, width)
        bottom = min(bottom, height)
        return left, top, right, bottom

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]
        aux_label = self.img_aux_labels[idx] if len(self.img_aux_labels) > 0 else None

        # Get patch size
        width, height = self.get_image_size(img_path)
        if self.is_crop and self.crop_range is not None:
            crop_box = self.adjust_crop_box(self.crop_range[idx], width, height)
        else:
            crop_box = (0, 0, width, height)
        left, top, right, bottom = crop_box

        # Load patch
        obj = self.get_image_object(img_path)
        if isinstance(obj, Image.Image):
            patch = obj.crop((left, top, right, bottom))
        else:
            patch = obj[top:bottom, left:right]
            if patch.dtype != np.uint8:
                if patch.max() <= 1.0:
                    patch = (patch * 255).astype(np.uint8)
                else:
                    patch = patch.astype(np.uint8)
            patch = Image.fromarray(patch, mode='RGB')

        # Apply transform
        if self.transform:
            patch = self.transform(patch)

        # Return
        return_dict = {
            "image": patch,
            "label": label,
        }

        if aux_label is not None:
            return_dict["aux_label"] = aux_label
        if self.return_region and self.crop_range is not None:
            return_dict["region"] = (left, top, right, bottom)

        return return_dict
            

class PatchGridDataset(_BaseLargeImageDataset):
    """
    Dataset for extracting fixed-size patches from large images using a sliding window strategy.

    Supports PNG, TIFF, and zarr formats.

    Args:
        img_names: List of large image file names or zarr folders.
        patch_size: Patch size (patch_size x patch_size).
        stride: Sliding window stride.
        target_labels: List of target labels corresponding to images.
        transform: Transform to apply to each extracted patch.
        dataset_dir: Directory containing `img_names`.
        return_region: Whether to return patch coordinates (left, top, right, bottom).

    Returns:
        A dictionary containing the extracted patch, label, and optional region.
    """
    def __init__(
        self,
        img_names,
        patch_size,
        stride,
        target_labels,
        transform=None,
        dataset_dir=None,
        return_region=False,
    ):
        self.img_names = img_names
        self.patch_size = patch_size
        self.stride = stride
        self.labels = target_labels
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.return_region = return_region

        self.patch_infos = []

        assert len(self.labels) == len(self.img_names), 'The number of labels and images is inconsistent.'

        self.process_with_labels()

    def process_with_labels(self):
        for img_name, label in zip(self.img_names, self.labels):
            if str(label) != 'nan' and label is not None:
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    img_path = os.path.join(self.dataset_dir, img_name)
                else:
                    # zarr folder
                    img_path = os.path.join(self.dataset_dir, img_name, "he_img")

                if not os.path.exists(img_path):
                    print(f"{img_path} doesn't exist and has been skipped.")
                    continue

                width, height = self.get_image_size(img_path)
                x_starts, y_starts = self.get_patch_starts(width, height, 
                                                           self.patch_size, 
                                                           self.stride)

                for left in x_starts:
                    for top in y_starts:
                        right = min(left + self.patch_size, width)
                        bottom = min(top + self.patch_size, height)
                        self.patch_infos.append(
                            (img_path, (left, top, right, bottom), label)
                        )

    def get_patch_starts(self, width, height, patch_size, stride):
        x_starts = np.arange(0, width, stride)
        y_starts = np.arange(0, height, stride)
        if (width - x_starts[-1]) < patch_size / 2:
            x_starts = x_starts[:-1]
        if (height - y_starts[-1]) < patch_size / 2:
            y_starts = y_starts[:-1]
        return x_starts, y_starts
    
    def __len__(self):
        return len(self.patch_infos)

    def __getitem__(self, idx):
        img_path, (left, top, right, bottom), label = self.patch_infos[idx]

        # Extract patch
        obj = self.get_image_object(img_path)
        if isinstance(obj, Image.Image):
            patch = obj.crop((left, top, right, bottom))
        else:
            patch = obj[top:bottom, left:right]
            if patch.dtype != np.uint8:
                if patch.max() <= 1.0:
                    patch = (patch * 255).astype(np.uint8)
                else:
                    patch = patch.astype(np.uint8)
            patch = Image.fromarray(patch, mode='RGB')

        # Apply transform
        if self.transform:
            patch = self.transform(patch)

        # Return
        return_dict = {
            "image": patch,
            "label": label,
        }

        if self.return_region:
            return_dict["region"] = (left, top, right, bottom)

        return return_dict

