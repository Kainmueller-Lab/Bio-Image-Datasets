from bio_image_datasets.dataset import Dataset


class SegPath(Dataset):
    def __init__(self, local_path: str):
        super().__init__(self, local_path: str)
        pass

    def download(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

    def get_he(self, idx):
        return None

    def get_semantic_mask(self, idx):
        return None

    def get_instance_mask(self, idx):
        return None

    def get_sample_name(self, idx):
        return None

    def get_sample_names(self):
        return None

    def __repr__(self):
        return None