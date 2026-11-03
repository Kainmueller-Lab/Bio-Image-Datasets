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


def prepare_pathocell2_registered_he_samples(output_dir, num_samples=4, size=(32, 32)):
    """Create mock TIFF files where the registered H&E images are the preferred default."""
    registered_he_dir = os.path.join(output_dir, "he_image_registered")
    if_dir = os.path.join(output_dir, "if_image")
    masks_dir = os.path.join(output_dir, "annotation", "dataset_versions", "v_0-0-1-beta", "masks")
    pheno_dir = os.path.join(output_dir, "annotation", "dataset_versions", "v_0-0-1-beta", "pheno")

    for directory in [registered_he_dir, if_dir, masks_dir, pheno_dir]:
        os.makedirs(directory, exist_ok=True)

    sample_names = [f"sample_{index}" for index in range(num_samples)]
    for sample_name in sample_names:
        registered_he_image = np.full((size[0], size[1], 3), 7, dtype=np.uint8)
        if_image = np.random.randint(0, 256, size=(size[0], size[1], 4), dtype=np.uint8)
        instance_mask = np.random.randint(0, 1000, size=size, dtype=np.uint16)
        semantic_mask = np.random.randint(0, 14, size=size, dtype=np.uint8)

        io.imsave(os.path.join(registered_he_dir, f"{sample_name}_he_registered.tif"), registered_he_image)
        io.imsave(os.path.join(if_dir, f"{sample_name}_if.tif"), if_image)
        io.imsave(os.path.join(masks_dir, f"{sample_name}_cell_masks.tif"), instance_mask)
        io.imsave(os.path.join(pheno_dir, f"{sample_name}_pheno.tif"), semantic_mask)

    return sample_names


def prepare_pathocell2_non_registered_he_override_samples(output_dir, num_samples=4, size=(32, 32)):
    """Create mock TIFF files where one sample explicitly opts back to the non-registered H&E."""
    registered_he_dir = os.path.join(output_dir, "he_image_registered")
    legacy_he_dir = os.path.join(output_dir, "he_image")
    if_dir = os.path.join(output_dir, "if_image")
    masks_dir = os.path.join(output_dir, "annotation", "dataset_versions", "v_0-0-1-beta", "masks")
    pheno_dir = os.path.join(output_dir, "annotation", "dataset_versions", "v_0-0-1-beta", "pheno")

    for directory in [registered_he_dir, legacy_he_dir, if_dir, masks_dir, pheno_dir]:
        os.makedirs(directory, exist_ok=True)

    sample_names = [f"sample_{index}" for index in range(num_samples)]
    for sample_name in sample_names:
        registered_he_image = np.full((size[0], size[1], 3), 7, dtype=np.uint8)
        legacy_he_image = np.full((size[0], size[1], 3), 9, dtype=np.uint8)
        if_image = np.random.randint(0, 256, size=(size[0], size[1], 4), dtype=np.uint8)
        instance_mask = np.random.randint(0, 1000, size=size, dtype=np.uint16)
        semantic_mask = np.random.randint(0, 14, size=size, dtype=np.uint8)

        io.imsave(os.path.join(registered_he_dir, f"{sample_name}_he_registered.tif"), registered_he_image)
        io.imsave(os.path.join(legacy_he_dir, f"{sample_name}_he_non_registered.tif"), legacy_he_image)
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
        assert "cell_mask" in sample
        assert "sample_name" in sample


def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        he_image = dataset.get_he(0)
        assert he_image.shape == (3, 32, 32)


def test_get_he_prefers_registered_images_by_default():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_registered_he_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        he_image = dataset.get_he(0)
        assert he_image.shape == (3, 32, 32)
        assert he_image.dtype == np.uint8
        assert np.all(he_image == 7)


def test_get_he_uses_original_he_for_explicit_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_names = prepare_pathocell2_non_registered_he_override_samples(tmp_dir, num_samples=4)
        original_he_dir = os.path.join(tmp_dir, 'he_image')
        os.makedirs(original_he_dir, exist_ok=True)
        for sample_name in sample_names[:3]:
            original_he_image = np.full((32, 32, 3), 9, dtype=np.uint8)
            io.imsave(os.path.join(original_he_dir, f'{sample_name}_he.tif'), original_he_image)

        original_samples = sample_names[:3]
        dataset = PathoCell2Dataset(local_path=tmp_dir, original_he_sample_names=original_samples)

        for idx, sample_name in enumerate(original_samples):
            he_image = dataset.get_he(dataset.get_sample_names().index(sample_name))
            assert np.all(he_image == 9)

        default_he_image = dataset.get_he(3)
        assert np.all(default_he_image == 7)


def test_get_he_uses_requested_original_sample_names_for_reg_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        registered_he_dir = os.path.join(tmp_dir, 'he_image_registered')
        original_he_dir = os.path.join(tmp_dir, 'he_image')
        if_dir = os.path.join(tmp_dir, 'if_image')
        masks_dir = os.path.join(tmp_dir, 'annotation', 'dataset_versions', 'v_0-0-1-beta', 'masks')
        pheno_dir = os.path.join(tmp_dir, 'annotation', 'dataset_versions', 'v_0-0-1-beta', 'pheno')
        for directory in [registered_he_dir, original_he_dir, if_dir, masks_dir, pheno_dir]:
            os.makedirs(directory, exist_ok=True)

        registered_he = np.full((32, 32, 3), 7, dtype=np.uint8)
        original_he = np.full((32, 32, 3), 9, dtype=np.uint8)
        if_image = np.random.randint(0, 256, size=(32, 32, 4), dtype=np.uint8)
        instance_mask = np.random.randint(0, 1000, size=(32, 32), dtype=np.uint16)
        semantic_mask = np.random.randint(0, 14, size=(32, 32), dtype=np.uint8)

        io.imsave(os.path.join(registered_he_dir, 'reg001_a_he_registered.tiff'), registered_he)
        io.imsave(os.path.join(original_he_dir, 'reg001_A_he.tiff'), original_he)
        io.imsave(os.path.join(if_dir, 'reg001_a_if.tiff'), if_image)
        io.imsave(os.path.join(masks_dir, 'reg001_a_cell_masks.tiff'), instance_mask)
        io.imsave(os.path.join(pheno_dir, 'reg001_a_pheno.tiff'), semantic_mask)

        dataset = PathoCell2Dataset(local_path=tmp_dir, original_he_sample_names=['reg001_A'])
        assert np.all(dataset.get_he(0) == 9)
        assert dataset.get_sample_name(0) == 'reg001_a'


def test_get_he_uses_original_for_all_samples_when_registered_false():
    with tempfile.TemporaryDirectory() as tmp_dir:
        registered_he_dir = os.path.join(tmp_dir, 'he_image_registered')
        original_he_dir = os.path.join(tmp_dir, 'he_image')
        if_dir = os.path.join(tmp_dir, 'if_image')
        masks_dir = os.path.join(tmp_dir, 'annotation', 'dataset_versions', 'v_0-0-1-beta', 'masks')
        pheno_dir = os.path.join(tmp_dir, 'annotation', 'dataset_versions', 'v_0-0-1-beta', 'pheno')
        for directory in [registered_he_dir, original_he_dir, if_dir, masks_dir, pheno_dir]:
            os.makedirs(directory, exist_ok=True)

        registered_he = np.full((32, 32, 3), 7, dtype=np.uint8)
        original_he = np.full((32, 32, 3), 9, dtype=np.uint8)
        if_image = np.random.randint(0, 256, size=(32, 32, 4), dtype=np.uint8)
        instance_mask = np.random.randint(0, 1000, size=(32, 32), dtype=np.uint16)
        semantic_mask = np.random.randint(0, 14, size=(32, 32), dtype=np.uint8)

        io.imsave(os.path.join(registered_he_dir, 'reg001_a_he_registered.tiff'), registered_he)
        io.imsave(os.path.join(original_he_dir, 'reg001_A_he.tiff'), original_he)
        io.imsave(os.path.join(if_dir, 'reg001_a_if.tiff'), if_image)
        io.imsave(os.path.join(masks_dir, 'reg001_a_cell_masks.tiff'), instance_mask)
        io.imsave(os.path.join(pheno_dir, 'reg001_a_pheno.tiff'), semantic_mask)

        dataset = PathoCell2Dataset(local_path=tmp_dir, registered=False)
        assert np.all(dataset.get_he(0) == 9)


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


def test_get_cell_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_pathocell2_samples(tmp_dir, num_samples=4)
        dataset = PathoCell2Dataset(local_path=tmp_dir)
        cell_mask = dataset.get_cell_mask(0)
        assert cell_mask is not None
        assert cell_mask.shape == (32, 32)


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