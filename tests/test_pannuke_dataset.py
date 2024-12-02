from bio_image_datasets.pannuke import PanNukeDataset, mapping_dict
import os
import h5py
import numpy as np
import tempfile


def prepare_pannuke_samples(output_dir, num_samples=5):
    """
    Creates mock files adhering to the PanNuke dataset specifications.

    Args:
        output_dir (str): Path to the directory where the mock files will be created.
        num_samples (int): Number of mock samples to create.

    Returns:
        list: List of paths to the created mock HDF files.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []

    pass


def test_len():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_pannuke_samples(tmp_dir, num_samples=5)
        dataset = PanNukeDataset(local_path=tmp_dir)
        assert len(dataset) == 5


def test_getitem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_pannuke_samples(tmp_dir, num_samples=5)
        dataset = PanNukeDataset(local_path=tmp_dir)
        sample = dataset[0]
        # TODO


def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_pannuke_samples(tmp_dir, num_samples=5)
        dataset = PanNukeDataset(local_path=tmp_dir)
        # TODO


def test_get_if():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_pannuke_samples(tmp_dir, num_samples=5)
        dataset = PanNukeDataset(local_path=tmp_dir)
        # TODO


def test_get_class_mapping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = PanNukeDataset(local_path=tmp_dir)
        class_mapping = dataset.get_class_mapping()
    assert class_mapping == mapping_dict


def test_get_instance_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_pannuke_samples(tmp_dir, num_samples=5)
        dataset = PanNukeDataset(local_path=tmp_dir)
        # TODO


def test_get_semantic_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_pannuke_samples(tmp_dir, num_samples=5)
        dataset = PanNukeDataset(local_path=tmp_dir)
        # TODO


def test_get_sample_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_pannuke_samples(tmp_dir, num_samples=5)
        dataset = PanNukeDataset(local_path=tmp_dir)
        # TODO


def test_get_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = prepare_pannuke_samples(tmp_dir, num_samples=5)
        dataset = PanNukeDataset(local_path=tmp_dir)
        pass