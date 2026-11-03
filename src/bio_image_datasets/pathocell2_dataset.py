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
    10: 'Smooth muscle Cells / Pericytes',
    11: 'Fibroblasts',
    12: 'Pericytes (outdated)',
    13: 'Dendritic Cells',
}


class PathoCell2Dataset(Dataset):
    def __init__(self, local_path, version='v_0-0-1-beta', original_he_sample_names=None, registered=True):
        """
        Initialize the PatchoCell2 dataset from TIFF files on disk.

        The preliminary masks are stored under:
        <local_path>/annotation/dataset_versions/<version>/masks
        and <local_path>/annotation/dataset_versions/<version>/pheno

        The original H&E images are stored under:
        <local_path>/he_image

        The registered H&E images are stored under:
        <local_path>/he_image_registered

        The mIF images are stored under:
        <local_path>/if_image

        Args:
            local_path (str): Root directory containing the dataset folders.
            version (str): Dataset version folder name under annotation/dataset_versions.
            original_he_sample_names (list[str] | None): Sample names that should use the
                original (non-registered) H&E image instead of the registered one.
            registered (bool): If True, use the registered H&E by default and only use the
                original H&E for explicit sample names in ``original_he_sample_names``.
                If False, use the original H&E for all samples.
        """
        super().__init__(local_path)
        self.local_path = os.path.expanduser(local_path)
        self.version = version
        self.registered = registered
        self.original_he_sample_names = {self._normalize_sample_name(name) for name in (original_he_sample_names or ())}

        self.he_registered_dir = os.path.join(self.local_path, 'he_image_registered')
        self.he_dir = os.path.join(self.local_path, 'he_image')
        self.he_dirs = self._resolve_he_dirs()
        self.if_dir = os.path.join(self.local_path, 'if_image')
        self.instance_dir = os.path.join(self.local_path, 'annotation', 'dataset_versions', self.version, 'masks')
        self.semantic_dir = os.path.join(self.local_path, 'annotation', 'dataset_versions', self.version, 'pheno')

        self.sample_names = self._discover_sample_names()
        if not self.sample_names:
            raise ValueError(
                "No matching TIFF samples were found in the expected directories. "
                f"Looked for files under: {self.he_dirs}, {self.if_dir}, {self.instance_dir}, and {self.semantic_dir}."
            )

    def _resolve_he_dirs(self):
        """Return the registered H&E directory first, then the original H&E directory as fallback."""
        he_dirs = []
        for directory_path in [self.he_registered_dir, self.he_dir]:
            if os.path.isdir(directory_path):
                he_dirs.append(directory_path)

        if not he_dirs:
            he_dirs.append(self.he_dir)
        return he_dirs

    def _discover_sample_names(self):
        """Collect sample names that exist in all required directories."""
        he_sample_stems = set()
        for directory in self.he_dirs:
            if not os.path.isdir(directory):
                continue
            for file_name in os.listdir(directory):
                if not file_name.lower().endswith(('.tif', '.tiff', '.png')):
                    continue
                stem = os.path.splitext(file_name)[0]
                sample_stem = self._sample_stem_from_file_name(stem)
                if sample_stem is not None:
                    he_sample_stems.add(sample_stem)

        directory_paths = [self.if_dir, self.instance_dir, self.semantic_dir]
        discovered = [he_sample_stems]
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
    def _normalize_sample_name(sample_name):
        """Normalize sample-name matching across case differences."""
        return sample_name.lower()

    @classmethod
    def _sample_stem_from_file_name(cls, file_name):
        """Normalize PathoCell2 H&E file names to a canonical sample identifier."""
        suffixes = [
            '_cell_masks', '_nuclei_masks', '_cell_mask', '_nuclei_mask',
            '_pheno', '_phenotype', '_mask', '_masks',
            '_he_registered', '_he', '_if', '_simple_cell_types'
        ]
        lowered_name = file_name.lower()
        for suffix in suffixes:
            if lowered_name.endswith(suffix):
                return cls._normalize_sample_name(file_name[:-len(suffix)])
        return cls._normalize_sample_name(file_name)

    def _get_sample_path(self, directory, sample_name):
        """Return the first matching file for a sample in the given directory priority order."""
        if isinstance(directory, (list, tuple)):
            directories = list(directory)
        else:
            directories = [directory]

        for current_directory in directories:
            for file_name in sorted(os.listdir(current_directory)):
                if file_name.lower().endswith(('.tif', '.tiff', '.png')):
                    stem = os.path.splitext(file_name)[0]
                    if self._sample_stem_from_file_name(stem) == sample_name:
                        return os.path.join(current_directory, file_name)

        search_dirs = ', '.join(str(path) for path in directories)
        raise FileNotFoundError(f"No file for sample '{sample_name}' found in {search_dirs}")

    def _get_he_path(self, sample_name):
        """Select the H&E image according to the registration mode and sample-specific overrides."""
        normalized_name = self._normalize_sample_name(sample_name)

        if self.registered:
            preferred_dirs = [self.he_registered_dir, self.he_dir]
            if normalized_name in self.original_he_sample_names and os.path.isdir(self.he_dir):
                preferred_dirs = [self.he_dir, self.he_registered_dir]
        else:
            preferred_dirs = [self.he_dir, self.he_registered_dir]

        preferred_dirs = [directory for directory in preferred_dirs if os.path.isdir(directory)]
        if not preferred_dirs:
            preferred_dirs = [self.he_dir]

        return self._get_sample_path(preferred_dirs, normalized_name)

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
        if directory == self.he_dirs:
            path = self._get_he_path(sample_name)
        else:
            path = self._get_sample_path(directory, sample_name)
        image = np.asarray(io.imread(path))

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
            'cell_mask': self.get_cell_mask(idx),
            'sample_name': sample_name,
        }

    def get_he(self, idx):
        """Load the H&E image at the given index."""
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        return self._load_image(self.he_dirs, idx)

    def get_if(self, idx):
        """Load the mIF image at the given index."""
        if idx >= len(self):
            raise IndexError("Index out of bounds.")
        return self._load_image(self.if_dir, idx)

    def get_class_mapping(self):
        """Return the class mapping for the dataset."""
        return coarse_mapping

    def get_instance_mask(self, idx):
        """Return the nuclei instance mask at the given index.

        In this dataset, instance files can be provided as separate cell and nuclei masks,
        so the nuclei mask is treated as the primary instance mask and the cell mask is
        exposed separately via get_cell_mask().
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds.")

        sample_name = self.sample_names[idx]
        _, nuclei_path = self._get_instance_mask_paths(sample_name)
        if nuclei_path is not None:
            return self._load_mask_from_path(nuclei_path)
        return self._load_mask(self.instance_dir, idx)

    def get_cell_mask(self, idx):
        """Return the cell instance mask at the given index if it exists."""
        if idx >= len(self):
            raise IndexError("Index out of bounds.")

        sample_name = self.sample_names[idx]
        cell_path, _ = self._get_instance_mask_paths(sample_name)
        if cell_path is None:
            return None
        return self._load_mask_from_path(cell_path)

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
        return f"PathoCell2Dataset ({self.local_path}, {len(self)} samples)"
