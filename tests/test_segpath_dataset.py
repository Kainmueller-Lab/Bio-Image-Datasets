from bio_image_datasets.segpath_dataset import SegPath
import os
from skimage import io
import numpy as np
import tempfile


def prepare_segpath_samples(output_dir, num_samples_per_folder=2, size=(64, 64)):
    """
    Creates mock .png files adhering to the SegPath dataset specifications.

    Args:
        output_dir (str): Path to the directory where the mock HDF files will be created.
        num_samples (int): Number of mock samples to create.
        num_folders (int): Number of folders with the images of each cell type
    Returns:
        list: List of paths to the created mock HDF files.
    """
    os.makedirs(output_dir, exist_ok=True)
    folder_names = sorted(['panCK_Epithelium', 'CD3CD20_Lymphocyte'])
    file_paths = []
    images = []
    masks = []
    for folder in folder_names:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
        for j in range(num_samples_per_folder):
            file_name = f"{folder}_{j}_HE.png"
            file_path = os.path.join(output_dir, folder, file_name)
            file_paths.append(file_path.replace('_HE.png', ''))
            image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
            mask = np.random.randint(0, 2, size=(64, 64), dtype=np.uint8)
            images.append(image)
            masks.append(mask)
            io.imsave(file_path, image, dtype=np.uint8)
            io.imsave(file_path.replace('_HE.png', '_mask.png'), mask, dtype=np.uint8)
    return folder_names, file_paths, images, masks, folder_names


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        folder_names, file_paths, _, _, _ = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir) 
        assert dataset.mapping_dict == {
            0: 'Background', 1: 'CD3CD20_Lymphocyte', 2: 'panCK_Epithelium'}
        assert dataset.ext_HE == '_HE.png'
        assert dataset.ext_sem_mask == '_mask.png'
        assert dataset.image_paths == file_paths


def test_len():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prep_out = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        assert len(dataset) == 4


def test_getitem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, images, masks, _ = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        he, semantic_mask = dataset[0]
        assert he.shape == (64, 64, 3)
        assert semantic_mask.shape == (64, 64)
        assert np.allclose(he, images[0])
        assert np.allclose(semantic_mask, masks[0])


def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, images, _, _ = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        he = dataset.get_he(0)
        assert he.shape == (64, 64, 3)
        assert np.allclose(he, images[0])


def test_get_semantic_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, _, masks, _ = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        assert dataset.get_semantic_mask(0).shape == (64, 64)
        assert np.allclose(dataset.get_semantic_mask(0), masks[0])
        assert np.unique(dataset.get_semantic_mask(-1)).tolist() == [0, 2]


def test_get_class_idx():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prep_out = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        class_idx = dataset._get_class_idx('panCK_Epithelium')
        assert class_idx == 2


def test_get_class_mapping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, _, _, cell_types = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        class_mapping = dataset.get_mapping_dict()
        gt_mapping = {0: 'Background', 1: 'CD3CD20_Lymphocyte', 2: 'panCK_Epithelium'}
    assert class_mapping == gt_mapping
