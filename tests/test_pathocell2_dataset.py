import os
import tempfile

import numpy as np
from skimage import io

from bio_image_datasets.pathocell2_dataset import PathoCell2Dataset, coarse_mapping


def prepare_pathocell2_samples(output_dir, num_samples=4, size=(32, 32)):
    """
    Creates mock TIFF files adhering to the PathoCell2 dataset layout.

    Args:
        output_dir (str): Path to the directory where the mock files will be created.
        num_samples (int): Number of mock samples to create.
        size (tuple): Height and width of the mock images and masks.

    Returns:
        list: The sample names used to create the mock files.
    """
    he_dir = os.path.join(output_dir, "he_image")
    if_dir = os.path.join(output_dir, "if_image")
    masks_dir = os.path.join(output_dir, "annotation", "dataset_versions", "v_0-0-1-beta", "masks")
    pheno_dir = os.path.join(output_dir, "annotation", "dataset_versions", "v_0-0-1-beta", "pheno")

    for directory in [he_dir, if_dir, masks_dir, pheno_dir]:
        os.makedirs(directory, exist_ok=True)

    sample_names = [f"sample_{index}" for index in range(num_samples)]
    for sample_name in sample_names:
        he_image = np.random.randint(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
        if_image = np.random.randint(0, 256, size=(size[0], size[1], 4), dtype=np.uint8)
        instance_mask = np.random.randint(0, 1000, size=size, dtype=np.uint16)
        semantic_mask = np.random.randint(0, 14, size=size, dtype=np.uint8)

        io.imsave(os.path.join(he_dir, f"{sample_name}_he.tif"), he_image)
        io.imsave(os.path.join(if_dir, f"{sample_name}_if.tif"), if_image)
        io.imsave(os.path.join(masks_dir, f"{sample_name}_cell_masks.tif"), instance_mask)
        io.imsave(os.path.join(pheno_dir, f"{sample_name}_pheno.tif"), semantic_mask)

    return sample_names


def test_len():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        assert len(dataset) == 4


def test_getitem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        sample = dataset[0]
        assert "image" in sample
        assert "if_image" in sample
        assert "semantic_mask" in sample
        assert "instance_mask" in sample
        assert "nuclei_mask" in sample
        assert "sample_name" in sample


def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        he_image = dataset.get_he(0)
        assert he_image.shape == (3, 32, 32)


def test_get_if():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        if_image = dataset.get_if(0)
        assert if_image.shape == (4, 32, 32)


def test_get_class_mapping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        class_mapping = dataset.get_class_mapping()
    assert class_mapping == coarse_mapping


def test_get_instance_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        instance_mask = dataset.get_instance_mask(0)
        assert instance_mask.shape == (32, 32)


def test_get_semantic_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        semantic_mask = dataset.get_semantic_mask(0)
        assert semantic_mask.shape == (32, 32)


def test_get_nuclei_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        assert dataset.get_nuclei_mask(0) is None


def test_get_sample_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_names = prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        assert dataset.get_sample_name(0) == sample_names[0]
        assert dataset.get_sample_name(-1) == sample_names[-1]


def test_get_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_names = prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        assert dataset.get_sample_names() == sample_names


def test_repr():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        assert "PathoCell2Dataset" in repr(dataset)