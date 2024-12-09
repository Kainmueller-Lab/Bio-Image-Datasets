from bio_image_datasets.pannuke_dataset import PanNukeDataset, mapping_dict
import os
import numpy as np
import tempfile


def prepare_pannuke_samples(output_dir, num_samples_per_fold=5):
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

    folds = [1, 2, 3]
    for fold in folds:
        os.makedirs(os.path.join(output_dir, f'fold{fold}/images/fold{fold}'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'fold{fold}/masks/fold{fold}'), exist_ok=True)

        images = (np.random.rand(num_samples_per_fold, 256, 256, 3)*255).astype(np.float64)
        types = np.tile(sample_tissue_types, num_samples_per_fold // len(sample_tissue_types) + 1)[:num_samples_per_fold]
        masks = np.random.randint(0, 3000, size = (num_samples_per_fold, 256, 256, 6)).astype(np.float64)

        np.save(os.path.join(output_dir, f'fold{fold}/images/fold{fold}/images.npy'), images)
        np.save(os.path.join(output_dir, f'fold{fold}/images/fold{fold}/types.npy'), types)
        np.save(os.path.join(output_dir, f'fold{fold}/masks/fold{fold}/masks.npy'), masks)


def test_len():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_pannuke_samples(local_path, num_samples_per_fold=5)
        dataset = PanNukeDataset(local_path=local_path)
        assert len(dataset) == 15

def test_get_length_per_fold():
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = os.path.join(tmp_dir)
            prepare_pannuke_samples(local_path, num_samples_per_fold=5)
            dataset = PanNukeDataset(local_path=local_path)
            folds = [1, 2, 3]
            for fold in folds:
                fold_length = dataset.get_length_per_fold(fold)
                assert fold_length == 5

def test_getitem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_pannuke_samples(local_path, num_samples_per_fold=5)
        dataset = PanNukeDataset(local_path=local_path)
        sample = dataset[0]
        assert "image" in sample
        assert "type" in sample
        assert "semantic_mask" in sample
        assert "instance_mask" in sample
        assert "sample_name" in sample


def test_get_he():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_pannuke_samples(local_path, num_samples_per_fold=5)
        dataset = PanNukeDataset(local_path=local_path)
        he_image = dataset.get_he(0)
        assert he_image.shape == (3, 256, 256)


def test_get_tissue_type():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_pannuke_samples(local_path, num_samples_per_fold=5)
        dataset = PanNukeDataset(local_path=local_path)
        tissue_type = dataset.get_tissue_type(0)
        assert tissue_type == "Breast"


def test_get_class_mapping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_pannuke_samples(local_path, num_samples_per_fold=5)
        dataset = PanNukeDataset(local_path=local_path)
        class_mapping = dataset.get_class_mapping()
    assert class_mapping == mapping_dict


def test_get_instance_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_pannuke_samples(local_path, num_samples_per_fold=5)
        dataset = PanNukeDataset(local_path=local_path)
        instace_mask = dataset.get_instance_mask(0)
        assert instace_mask.shape == (256, 256)


def test_get_semantic_mask():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_pannuke_samples(local_path, num_samples_per_fold=5)
        dataset = PanNukeDataset(local_path=local_path)
        semantic_mask = dataset.get_semantic_mask(0)
        assert semantic_mask.shape == (256, 256)


def test_get_sample_name():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_pannuke_samples(local_path, num_samples_per_fold=5)
        dataset = PanNukeDataset(local_path=local_path)
        sample_name_f1 = dataset.get_sample_name(0)
        sample_name_f2 = dataset.get_sample_name(5)
        sample_name_f3 = dataset.get_sample_name(10)
        assert sample_name_f1 == "fold1_0"
        assert sample_name_f2 == "fold2_0"
        assert sample_name_f3 == "fold3_0"

def test_get_sample_names():
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir)
        prepare_pannuke_samples(local_path, num_samples_per_fold=5)
        dataset = PanNukeDataset(local_path=local_path)
        sample_names = dataset.get_sample_names()
        assert sample_names == ["fold1_0", "fold1_1", "fold1_2", "fold1_3", "fold1_4", "fold2_0", "fold2_1", "fold2_2", "fold2_3", "fold2_4", "fold3_0", "fold3_1", "fold3_2", "fold3_3", "fold3_4"]