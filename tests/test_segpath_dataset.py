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
            image = np.random.randint(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
            mask = np.random.randint(0, 2, size=(size[0], size[1]), dtype=np.uint8)
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
        width = round(64 * dataset.original_microns_per_pixel / dataset.microns_per_pixel)
        assert he.shape == (width, width, 3)
        assert semantic_mask.shape == (width, width)

        # without rescaling
        dataset.original_microns_per_pixel = 0.5
        he, semantic_mask = dataset[0]
        assert np.allclose(he, images[0])
        assert np.allclose(semantic_mask, masks[0])


def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, images, _, _ = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        he = dataset.get_he(0)
        width = round(64 * dataset.original_microns_per_pixel / dataset.microns_per_pixel)
        assert he.shape == (width, width, 3)

        # without rescaling
        dataset.original_microns_per_pixel = 0.5
        he = dataset.get_he(0)
        assert np.allclose(he, images[0])


def test_get_semantic_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, _, masks, _ = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        width = round(64 * dataset.original_microns_per_pixel / dataset.microns_per_pixel)
        assert dataset.get_semantic_mask(0).shape == (width, width)
        assert np.unique(dataset.get_semantic_mask(-1)).tolist() == [0, 2]

        # without rescaling
        dataset.original_microns_per_pixel = 0.5
        assert np.allclose(dataset.get_semantic_mask(0), masks[0])


def test_get_class_mapping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, _, _, _, cell_types = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        class_mapping = dataset.get_mapping_dict()
        gt_mapping = {0: 'Background', 1: 'CD3CD20_Lymphocyte', 2: 'panCK_Epithelium'}
    assert class_mapping == gt_mapping


def test_resize():
    with tempfile.TemporaryDirectory() as tmp_dir:
        width = 15
        _, _, images, masks, _ = prepare_segpath_samples(tmp_dir, size=(width, width))
        dataset = SegPath(local_path=tmp_dir, microns_per_pixel=0.5)
        
        # upsampling
        dataset.original_microns_per_pixel = 2
        new_width = round(width * dataset.original_microns_per_pixel / dataset.microns_per_pixel)
        image = images[0]
        mask = masks[0]
        image_resized = dataset._resize(image)
        mask_resized = dataset._resize(mask, interp='nearest')
        assert image_resized.shape == (new_width, new_width, 3)	
        assert mask_resized.shape == (new_width, new_width)
        assert np.allclose(mask_resized[::4, ::4], mask)
        assert np.unique(mask_resized).tolist() == [0, 1]

        # downsampling
        dataset.original_microns_per_pixel = 0.25
        new_width = round(width * dataset.original_microns_per_pixel / dataset.microns_per_pixel)
        image_resized = dataset._resize(image)
        mask_resized = dataset._resize(mask, interp='nearest')
        assert image_resized.shape == (new_width, new_width, 3)
        assert mask_resized.shape == (new_width, new_width)
        assert np.allclose(mask_resized, mask[::2, ::2])


def test_get_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, file_paths, _, _, _ = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)
        sample_names = [os.path.split(p)[1] for p in file_paths]
        assert dataset.get_sample_names() == sample_names


def test_get_sample_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, file_paths, _, _, _ = prepare_segpath_samples(tmp_dir)
        dataset = SegPath(local_path=tmp_dir)  
        assert dataset.get_sample_name(0) == os.path.split(file_paths[0])[1]
        assert dataset.get_sample_name(-1) == os.path.split(file_paths[-1])[1]