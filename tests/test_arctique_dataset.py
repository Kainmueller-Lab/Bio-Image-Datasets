from bio_image_datasets.arctique_dataset import ArctiqueDataset, mapping_dict
import os
import numpy as np
import tempfile
from PIL import Image


def prepare_arctique_samples(output_dir, num_samples=5):
    """
    Creates mock files adhering to the Arctique dataset specifications.

    Args:
        output_dir (str): Path to the directory where the mock files will be created.
        num_samples (int): Number of mock samples to create.

    Returns:
        list: List of paths to the created mock files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for folder_name in ["images", "masks/semantic_noise_0", "masks/instance_noise_0"]:
        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
    
    for i in range(num_samples):
        img_name = f"img_{i}.png"
        tmp_img_path = os.path.join(output_dir, img_name)
        mock_image = Image.new('RGB', (512, 512), color=(255, 0, 0))

        mask_name = f"{i}.png"
        instances = np.random.randint(0, 100, size=(512, 512), dtype=np.uint16)
        mock_mask_instances = Image.fromarray(instances)
        mock_mask_instances.save(os.path.join(output_dir, "masks/instance_noise_0", mask_name))
                        
        classes = np.random.randint(0, 6, size=(512, 512), dtype=np.uint16)
        mock_mask_semantic = Image.fromarray(classes)
        mock_mask_semantic.save(os.path.join(output_dir, "masks/semantic_noise_0", mask_name))

def test_len():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_arctique_samples(local_path, num_samples=5)
        dataset = ArctiqueDataset(local_path=local_path)
        assert len(dataset) == 5


def test_getitem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_arctique_samples(local_path)
        dataset = ArctiqueDataset(local_path=local_path)
        sample = dataset[0]
        assert "image" in sample
        assert "semantic_mask" in sample
        assert "instance_mask" in sample
        assert "sample_name" in sample


def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_arctique_samples(local_path)
        dataset = ArctiqueDataset(local_path=local_path)
        he_image = dataset.get_he(0)
        assert he_image.shape == (3, 512, 512)


def test_get_class_mapping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_arctique_samples(local_path)
        dataset = ArctiqueDataset(local_path=local_path)
        class_mapping = dataset.get_class_mapping()
    assert class_mapping == mapping_dict


def test_get_instance_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_arctique_samples(local_path)
        dataset = ArctiqueDataset(local_path=local_path)
        instace_mask = dataset.get_instance_mask(0)
        assert instace_mask.shape == (512, 512)


def test_get_semantic_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_arctique_samples(local_path)
        dataset = ArctiqueDataset(local_path=local_path)
        semantic_mask = dataset.get_semantic_mask(0)
        assert semantic_mask.shape == (512, 512)


def test_get_sample_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_arctique_samples(local_path)
        dataset = ArctiqueDataset(local_path=local_path)
        sample_name = dataset.get_sample_name(0)
        assert type(sample_name) == "str"


def test_get_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_arctique_samples(local_path)
        dataset = ArctiqueDataset(local_path=local_path)
        sample_names = dataset.get_sample_names()
        assert len(sample_names) == 5