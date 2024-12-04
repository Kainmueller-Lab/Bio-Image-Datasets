from bio_image_datasets.schuerch_dataset import (SchuerchDataset, mapping_dict, exclude_classes,
                                                 transform_semantic_mask)
import os
import h5py
import numpy as np
import tempfile


def prepare_schuerch_samples(output_dir, num_samples=5):
    """
    Creates mock HDF files adhering to the Schuerch dataset specifications.

    Args:
        output_dir (str): Path to the directory where the mock HDF files will be created.
        num_samples (int): Number of mock samples to create.

    Returns:
        list: List of paths to the created mock HDF files.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []

    for i in range(num_samples):
        file_name = f"sample_{i+1}.h5"
        file_path = os.path.join(output_dir, file_name)
        file_paths.append(file_path)

        with h5py.File(file_path, "w") as f:
            # Create mock data
            f.create_dataset(
                "gt_ct", data=np.random.randint(0, 30, size=(1, 1440, 1920), dtype=np.uint16)
            )
            f.create_dataset(
                "gt_inst", data=np.random.randint(0, 1000, size=(1, 1440, 1920), dtype=np.uint16)
            )
            f.create_dataset(
                "ifl", data=np.random.randint(0, 65535, size=(58, 1440, 1920), dtype=np.uint16)
            )
            f.create_dataset(
                "img", data=np.random.randint(0, 65535, size=(3, 1440, 1920), dtype=np.uint16)
            )
    
    return file_paths


def test_len():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_schuerch_samples(tmp_dir, num_samples=5)
        dataset = SchuerchDataset(local_path=tmp_dir)
        assert len(dataset) == 5


def test_getitem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_schuerch_samples(tmp_dir, num_samples=5)
        dataset = SchuerchDataset(local_path=tmp_dir)
        sample = dataset[0]
        assert "gt_ct" in sample
        assert "gt_inst" in sample
        assert "immunoflourescence_img" in sample
        assert "he_img" in sample


def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_schuerch_samples(tmp_dir, num_samples=5)
        dataset = SchuerchDataset(local_path=tmp_dir)
        he_data = dataset.get_he(0)
        assert he_data.shape == (3, 1440, 1920)


def test_get_if():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_schuerch_samples(tmp_dir, num_samples=5)
        dataset = SchuerchDataset(local_path=tmp_dir)
        if_data = dataset.get_if(0)
        assert if_data.shape == (58, 1440, 1920)


def test_get_class_mapping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_schuerch_samples(tmp_dir, num_samples=1)
        dataset = SchuerchDataset(local_path=tmp_dir)
        class_mapping = dataset.get_class_mapping()
    assert class_mapping == mapping_dict


def test_get_instance_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_schuerch_samples(tmp_dir, num_samples=5)
        dataset = SchuerchDataset(local_path=tmp_dir)
        instance_mask = dataset.get_instance_mask(0)
        assert instance_mask.shape == (1440, 1920)


def test_get_semantic_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_schuerch_samples(tmp_dir, num_samples=5)
        dataset = SchuerchDataset(local_path=tmp_dir)
        semantic_mask = dataset.get_semantic_mask(0)
        assert semantic_mask.shape == (1440, 1920)


def test_get_sample_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_schuerch_samples(tmp_dir, num_samples=5)
        dataset = SchuerchDataset(local_path=tmp_dir)
        sample_name = dataset.get_sample_name(0)
        assert sample_name in [f"sample_{i}.h5" for i in range(1, 6)]


def test_get_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_schuerch_samples(tmp_dir, num_samples=5)
        dataset = SchuerchDataset(local_path=tmp_dir)
        sample_names = dataset.get_sample_names()
        assert len(sample_names) == 5
        assert all([name in [f"sample_{i}.h5" for i in range(1, 6)] for name in sample_names])


def test_exclude_classes():
    # with semantic mask only
    semantic_mask = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    exclude_classes_list = [2, 5, 8]
    expected_semantic_mask = np.array([[1, 0, 3], [4, 0, 6], [7, 0, 9]])
    
    updated_semantic_mask = exclude_classes(semantic_mask, exclude_classes_list)
    np.testing.assert_array_equal(updated_semantic_mask, expected_semantic_mask)

    # with semantic and instance mask
    instance_mask = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    semantic_mask = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    exclude_classes_list = [2, 5, 8]
    expected_instance_mask = np.array([[9, 0, 7], [6, 0, 4], [3, 0, 1]])
    
    updated_semantic_mask, updated_instance_mask = exclude_classes(
        semantic_mask, exclude_classes_list, instance_mask
    )
    np.testing.assert_array_equal(updated_semantic_mask, expected_semantic_mask)
    np.testing.assert_array_equal(updated_instance_mask, expected_instance_mask)


def test_transform_semantic_mask():
    # transform semantic labels using a mapping dictionary
    semantic_mask = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    mapping_dict = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18}
    expected_transformed_mask = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    
    transformed_mask = transform_semantic_mask(semantic_mask, mapping_dict)
    np.testing.assert_array_equal(transformed_mask, expected_transformed_mask)
