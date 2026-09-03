import os
import re

import numpy as np
from skimage import io

from bio_image_datasets.dataset import Dataset


coarse_mapping = {
    0: 'Background',
    1: 'Tumor cells',
    2: 'T Cells',
    3: 'B Cells',
    4: 'Plasma Cells',
    5: 'NK Cells',
    6: 'Macrophages',
    7: 'Monocytes',
    8: 'Granulocytes',
    9: 'Endothelial Cells',
    10: 'Smooth muscle Cells',
    11: 'Fibroblasts',
    12: 'Pericytes',
    13: 'Dendritic Cells',
}


class PatchoCell2Dataset(Dataset):
    def __init__(self, local_path, version='v_0-0-1-beta'):
        """
        Initialize the PatchoCell2 dataset from TIFF files on disk.

        The preliminary masks are stored under:
        <local_path>/annotation/dataset_versions/<version>/masks
        and <local_path>/annotation/dataset_versions/<version>/pheno

        The H&E images are stored under:
        <local_path>/he_image

        The mIF images are stored under:
        <local_path>/if_image

        Args:
            local_path (str): Root directory containing the dataset folders.
            version (str): Dataset version folder name under annotation/dataset_versions.
        """
        super().__init__(local_path)
        self.local_path = os.path.expanduser(local_path)
        self.version = version

        self.he_dir = os.path.join(self.local_path, 'he_image')
        self.if_dir = os.path.join(self.local_path, 'if_image')
        self.instance_dir = os.path.join(self.local_path, 'annotation', 'dataset_versions', self.version, 'masks')
        self.semantic_dir = os.path.join(self.local_path, 'annotation', 'dataset_versions', self.version, 'pheno')

        self.sample_names = self._discover_sample_names()
        if not self.sample_names:
            raise ValueError(
                "No matching TIFF samples were found in the expected directories. "
                f"Looked for files under: {self.he_dir}, {self.if_dir}, {self.instance_dir}, and {self.semantic_dir}."
            )

    def _discover_sample_names(self):
        """Collect sample names that exist in all required directories."""
        directory_paths = [self.he_dir, self.if_dir, self.instance_dir, self.semantic_dir]
        discovered = []
        for directory in directory_paths:
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Required directory not found: {directory}")

            sample_stems = set()
            for file_name in os.listdir(directory):
                if not file_name.lower().endswith(('.tif', '.tiff', '.png')):
                    continue
                stem = os.path.splitext(file_name)[0]
                sample_stem = self._sample_stem_from_file_name(stem)
                if sample_stem is not None:
                    sample_stems.add(sample_stem)
            discovered.append(sample_stems)

        common_names = sorted(set.intersection(*discovered))
        return common_names

    @staticmethod
    def _sample_stem_from_file_name(file_name):
        """Normalize sample names from different PatchoCell2 file naming conventions."""
        suffixes = [
            '_cell_masks', '_nuclei_masks', '_cell_mask', '_nuclei_mask',
            '_pheno', '_phenotype', '_mask', '_masks', '_he', '_if'
        ]
        for suffix in suffixes:
            if file_name.endswith(suffix):
                return file_name[:-len(suffix)]
        return file_name

    def _get_sample_path(self, directory, sample_name):
        """Return the first matching file for a sample in a directory."""
        candidates = []
        for file_name in os.listdir(directory):
            if file_name.lower().endswith(('.tif', '.tiff', '.png')):
                stem = os.path.splitext(file_name)[0]
                if self._sample_stem_from_file_name(stem) == sample_name:
                    candidates.append(os.path.join(directory, file_name))

        if not candidates:
            raise FileNotFoundError(f"No file for sample '{sample_name}' found in {directory}")

        return sorted(candidates)[0]

    def _get_instance_mask_paths(self, sample_name):
        """Return both cell and nuclei instance masks for a sample, if present."""
        cell_path = None
        nuclei_path = None
        for file_name in os.listdir(self.instance_dir):
            if not file_name.lower().endswith(('.tif', '.tiff', '.png')):
                continue
            stem = os.path.splitext(file_name)[0]
            if self._sample_stem_from_file_name(stem) != sample_name:
                continue
            if file_name.lower().endswith('_cell_masks.tif') or file_name.lower().endswith('_cell_masks.tiff') or file_name.lower().endswith('_cell_mask.tif') or file_name.lower().endswith('_cell_mask.tiff'):
                cell_path = os.path.join(self.instance_dir, file_name)
            elif file_name.lower().endswith('_nuclei_masks.tif') or file_name.lower().endswith('_nuclei_masks.tiff') or file_name.lower().endswith('_nuclei_mask.tif') or file_name.lower().endswith('_nuclei_mask.tiff'):
                nuclei_path = os.path.join(self.instance_dir, file_name)

        return cell_path, nuclei_path

    def _load_image(self, directory, idx):
        """Load an image from TIFF and return it in channel-first layout."""
        sample_name = self.sample_names[idx]
        image = np.asarray(io.imread(self._get_sample_path(directory, sample_name)))

        if image.ndim == 2:
            return image[np.newaxis, :, :]
        if image.ndim == 3 and image.shape[-1] <= 16:
            return np.moveaxis(image, -1, 0)
        return image

    def _load_mask(self, directory, idx):
        """Load a mask from TIFF and ensure it is a 2D array."""
        sample_name = self.sample_names[idx]
        mask = np.asarray(io.imread(self._get_sample_path(directory, sample_name)))
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        return mask

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sample_names)

    def __getitem__(self, idx):
        """Return a sample as a dictionary at the given index."""
        if idx >= len(self):
            raise IndexError("Index out of bounds.")

        sample_name = self.sample_names[idx]
        return {
            'image': self.get_he(idx),
            'if_image': self.get_if(idx),
            'semantic_mask': self.get_semantic_mask(idx),
            'instance_mask': self.get_instance_mask(idx),
            'nuclei_mask': self.get_nuclei_mask(idx),
            'sample_name': sample_name,
        }

    def get_he(self, idx):
        """Load the H&E image at the given index."""
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        return self._load_image(self.he_dir, idx)

    def get_if(self, idx):
        """Load the mIF image at the given index."""
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        return self._load_image(self.if_dir, idx)

    def get_class_mapping(self):
        """Return the class mapping for the dataset."""
        return coarse_mapping

    def get_instance_mask(self, idx):
        """Return the cell instance mask at the given index.

        In this dataset, instance files can be provided as separate cell and nuclei masks,
        so the cell mask is treated as the primary instance mask and the nuclei mask is
        exposed separately via get_nuclei_mask().
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")

        sample_name = self.sample_names[idx]
        cell_path, _ = self._get_instance_mask_paths(sample_name)
        if cell_path is not None:
            return self._load_mask_from_path(cell_path)
        return self._load_mask(self.instance_dir, idx)

    def get_nuclei_mask(self, idx):
        """Return the nuclei instance mask at the given index if it exists."""
        if idx >= len(self):
            raise IndexError("Index out of bounds.")

        sample_name = self.sample_names[idx]
        _, nuclei_path = self._get_instance_mask_paths(sample_name)
        if nuclei_path is None:
            return None
        return self._load_mask_from_path(nuclei_path)

    def _load_mask_from_path(self, path):
        """Load a single mask from a TIFF path and sanitize its dimensionality."""
        mask = np.asarray(io.imread(path))
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        return mask

    def get_semantic_mask(self, idx):
        """Return the semantic cell type mask at the given index."""
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        return self._load_mask(self.semantic_dir, idx)

    def get_sample_name(self, idx):
        """Return the sample name for the given index."""
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        return self.sample_names[idx]

    def get_sample_names(self):
        """Return the list of all sample names."""
        return self.sample_names

    def __repr__(self):
        """Return the string representation of the dataset."""
        return f"PatchoCell2Dataset ({self.local_path}, {len(self)} samples)"
