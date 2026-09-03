import numpy as np
from skimage import io

from bio_image_datasets.pathocell2_dataset import PatchoCell2Dataset


def test_patchocell2_dataset_reads_tiff_directories(tmp_path):
    base = tmp_path / "phenocell" / "1-1"
    he_dir = base / "he_image"
    if_dir = base / "if_image"
    mask_dir = base / "annotation" / "dataset_versions" / "v_0-0-1-beta" / "masks"
    pheno_dir = base / "annotation" / "dataset_versions" / "v_0-0-1-beta" / "pheno"

    for directory in [he_dir, if_dir, mask_dir, pheno_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    he = np.zeros((32, 32, 3), dtype=np.uint8)
    if_image = np.zeros((32, 32, 5), dtype=np.uint8)
    cell_mask = np.zeros((32, 32), dtype=np.uint16)
    cell_mask[5:15, 5:15] = 1
    nuclei_mask = np.zeros((32, 32), dtype=np.uint16)
    nuclei_mask[8:12, 8:12] = 1
    semantic_mask = np.zeros((32, 32), dtype=np.uint8)
    semantic_mask[5:15, 5:15] = 2

    io.imsave(he_dir / "reg070_B_he.tiff", he)
    io.imsave(if_dir / "reg070_B_if.tiff", if_image)
    io.imsave(mask_dir / "reg070_B_cell_masks.tiff", cell_mask)
    io.imsave(mask_dir / "reg070_B_nuclei_masks.tiff", nuclei_mask)
    io.imsave(pheno_dir / "reg070_B_pheno.tiff", semantic_mask)

    dataset = PatchoCell2Dataset(str(base))

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["sample_name"] == "reg070_B"
    assert sample["image"].shape == (3, 32, 32)
    assert sample["if_image"].shape == (5, 32, 32)
    assert sample["instance_mask"].shape == (32, 32)
    assert sample["nuclei_mask"].shape == (32, 32)
    assert sample["semantic_mask"].shape == (32, 32)
    assert dataset.get_class_mapping()[2] == "T Cells"
