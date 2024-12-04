from bio_image_datasets.pannuke import PanNukeDataset, mapping_dict
import os
import numpy as np
import tempfile


def prepare_pannuke_samples(output_dir, num_samples=5):
    """
    Creates mock files adhering to the PanNuke dataset specifications.

    Args:
        output_dir (str): Path to the directory where the mock files will be created.
        num_samples (int): Number of mock samples to create.

    Returns:
        list: List of paths to the created mock files.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_tissue_types = [np.str_("Breast"), np.str_("Colon"), np.str_("Bile-duct"), np.str_("Esophagus")]

    image_path = 'images/fold1'
    masks_path = 'masks/fold1'

    images_file_name = "images.npy"
    types_file_name = "types.npy"
    masks_file_name = "masks.npy"

    images = (np.random.rand(num_samples, 256, 256, 3)*255).astype(np.float64)
    # fill types with num_sample elements from sample_tissue_types 
    types = np.tile(sample_tissue_types, num_samples // len(sample_tissue_types) + 1)[:num_samples]
    masks = np.random.randint(0, 3000, size = (num_samples, 256, 256, 6)).astype(np.float64)

    os.makedirs(os.path.join(output_dir, image_path), exist_ok=True)
    os.makedirs(os.path.join(output_dir, masks_path), exist_ok=True)

    np.save(os.path.join(output_dir, image_path, images_file_name), images)
    np.save(os.path.join(output_dir, image_path, types_file_name), types)
    np.save(os.path.join(output_dir, masks_path, masks_file_name), masks) 
    
    return images_file_name, types_file_name, masks_file_name


def test_len():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, 'fold1')
        file_paths = prepare_pannuke_samples(local_path, num_samples=5)
        dataset = PanNukeDataset(local_path=local_path)
        assert len(dataset) == 5


def test_getitem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, 'fold1')
        prepare_pannuke_samples(local_path, num_samples=5)
        dataset = PanNukeDataset(local_path=local_path)
        sample = dataset[0]
        assert "image" in sample
        assert "type" in sample
        assert "semantic_mask" in sample
        assert "instance_mask" in sample
        assert "sample_name" in sample


def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, 'fold1')
        prepare_pannuke_samples(local_path, num_samples=5)
        dataset = PanNukeDataset(local_path=local_path)
        he_image = dataset.get_he(0)
        assert he_image.shape == (3, 256, 256)


def test_get_tissue_type():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, 'fold1')
        prepare_pannuke_samples(local_path, num_samples=5)
        dataset = PanNukeDataset(local_path=local_path)
        tissue_type = dataset.get_tissue_type(0)
        assert tissue_type == "Breast"


def test_get_class_mapping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, 'fold1')
        prepare_pannuke_samples(local_path, num_samples=5)
        dataset = PanNukeDataset(local_path=local_path)
        class_mapping = dataset.get_class_mapping()
    assert class_mapping == mapping_dict


def test_get_instance_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, 'fold1')
        prepare_pannuke_samples(local_path, num_samples=5)
        dataset = PanNukeDataset(local_path=local_path)
        instace_mask = dataset.get_instance_mask(0)
        assert instace_mask.shape == (256, 256)


def test_get_semantic_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, 'fold1')
        prepare_pannuke_samples(local_path, num_samples=5)
        dataset = PanNukeDataset(local_path=local_path)
        semantic_mask = dataset.get_semantic_mask(0)
        assert semantic_mask.shape == (256, 256)


def test_get_sample_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, 'fold1')
        file_paths = prepare_pannuke_samples(local_path, num_samples=5)
        dataset = PanNukeDataset(local_path=local_path)
        sample_name = dataset.get_sample_name(0)
        assert sample_name == "0"

def test_get_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, 'fold1')
        file_paths = prepare_pannuke_samples(local_path, num_samples=5)
        dataset = PanNukeDataset(local_path=local_path)
        sample_names = dataset.get_sample_names()
        assert sample_names == ["0", "1", "2", "3", "4"]