import os
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import scipy.io as sio

from bio_image_datasets.consep_dataset import ConSePDataset



def prepare_consep_samples(output_dir, num_samples=5):
    """
    #Creates mock  files adhering to the Schuerch dataset specifications.

    Args:
        output_dir (str): Path to the directory where the mock HDF files will be created.
        num_samples (int): Number of mock samples to create.

    Returns:
        list: List of paths to the created mock HDF files.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_path = output_dir.joinpath("tiles")
    label_path = output_dir.joinpath("labels")
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)


    file_paths = [f"sample{i}.png" for i in range(num_samples)]

    for filename in file_paths:

        tmp_image_path = os.path.join(image_path, filename)
        mock_image = Image.new('RGB', (244, 244), color=(255, 0, 0))
        mock_image.save(tmp_image_path)

        instance_map = np.random.randint(0, 10, size=(244, 244), dtype=np.uint16)
        class_labels = np.random.randint(0, 3, size=(10), dtype=np.uint16)
        empty_array = np.zeros(10)

        mock_label_data = {'__header__':None, 
                           '__version__':None, 
                           '__globals__':None, 
                           'roi_name':empty_array, 
                           'has_box':empty_array, 
                           'instance_map':instance_map, 
                           'has_centroid':empty_array, 
                           'has_label':empty_array, 
                           'n_cells_str':empty_array, 
                           'has_mask':empty_array, 
                           'fully_annotated_str':empty_array, 
                           'class_labels':class_labels, 
                           'coordinates':empty_array}

        sio.savemat(label_path.joinpath(filename.replace(".png", ".mat")), mock_label_data)

    return file_paths


def test_len():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_consep_samples(tmp_dir, num_samples=5)
        dataset = ConSePDataset(local_path=tmp_dir)
        assert len(dataset) == 5


def test_getitem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_consep_samples(tmp_dir, num_samples=1)
        dataset = ConSePDataset(local_path=tmp_dir)
        sample = dataset[0]
        assert "image" in sample
        assert "semantic_mask" in sample
        assert "instance_mask" in sample
        assert "sample_name" in sample
   

def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_consep_samples(tmp_dir, num_samples=1)
        dataset = ConSePDataset(local_path=tmp_dir)
        he_data = dataset.get_he(0)
        assert he_data.shape == (3, 244, 244)


def test_get_instance_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_consep_samples(tmp_dir, num_samples=5)
        dataset = ConSePDataset(local_path=tmp_dir)
        instance_mask = dataset.get_instance_mask(0)
        assert instance_mask.shape == (244, 244)


def test_get_semantic_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_consep_samples(tmp_dir, num_samples=5)
        dataset = ConSePDataset(local_path=tmp_dir)
        semantic_mask = dataset.get_semantic_mask(0)
        assert semantic_mask.shape == (244, 244)


def test_get_sample_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_consep_samples(tmp_dir, num_samples=5)
        dataset = ConSePDataset(local_path=tmp_dir)
        sample_name = dataset.get_sample_name(0)
        assert sample_name in [f"sample_{i}.png" for i in range(1, 6)]


def test_get_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_consep_samples(tmp_dir, num_samples=5)
        dataset = ConSePDataset(local_path=tmp_dir)
        sample_names = dataset.get_sample_names()
        assert len(sample_names) == 5
        assert all([name in [f"sample_{i}.h5" for i in range(1, 6)] for name in sample_names])