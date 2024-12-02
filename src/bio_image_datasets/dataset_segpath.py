import os
from skimage import io
from bio_image_datasets.dataset import Dataset



class SegPath(Dataset):
    def __init__(self, local_path: str):
        super().__init__(self, local_path)
        self.local_path = local_path
        self.image_paths, self.mask_paths = self.load_data(cell_types=['LYM'])

    def load_data(self, cell_types: list = ['EPI', 'LYM', 'MUS', 'RBC']):
        """Load the data paths for the HE images and the semantic masks.
        Args:
            cell_types (list): List of cell types to load. Default is ['EPI', 'LYM', 'MUS'].
        Returns:
            image_path (list): List of paths to the HE images.
            mask_path (list): List of paths to the semantic masks
        """
        self.classname2celltype = {
            'EPI': 'panCK_Epithelium',
            'LYM': 'CD3CD20_Lymphocyte',
            'MUS': 'aSMA_SmoothMuscle',
            'RBC': 'CD235a_RBC',
            'LEU': 'CD45RB_Leukocyte',
            'END': 'ERG_Endothelium',
            'PLA': 'MIST1_PlasmaCell',
            'MYE': 'MNDA_MyeloidCell'}  # folder names
        self.classes = {
            idx + 1: [cell_type, self.classname2celltype[cell_type]] 
            for idx, cell_type in enumerate(cell_types)}
        self.classes[0] = ['BG', 'Background']
        ext_HE = '_HE.png'
        ext_sem_mask = '_mask.png'
        for cell_type in cell_types:
            path = os.path.join(self.local_path, self.classname2celltype[cell_type])
            files = os.listdir(path)
            image_paths = [os.path.join(path, f) for f in files if f.endswith(ext_HE)]
            mask_paths = [os.path.join(path, f) for f in files if f.endswith(ext_sem_mask)]
        return image_paths, mask_paths
    
    def _get_cell_type(self, path_image):
        """Get the cell type from the image path.
        Args:
            path_image (str): Path to the image.
        Returns:
            cell_type (str): Cell type.
        """
        return path_image.split('/')[-1]

    def _get_class_idx(self, cell_type):
        """Get the class index from the cell type.
        Args:
            specific_cell_type (str): Specific cell type.
        Returns:
            class_idx (int): Class index.
        """
        for key, value in self.classes.items():
            if value[-1] == cell_type:
                return key
        return 0

    def download(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        """Fetches the HE image and the semantic mask at the given index.
        Args:
            idx (int): Index of the sample.
        Returns:
            he (np.array): HE image.
            semantic_mask (np.array): Semantic mask.
        """
        he = self.get_he(idx)
        semantic_mask = self.get_semantic_mask(idx)
        return he, semantic_mask

    def get_he(self, idx):
        """Return the HE image at the given index.
        Args:
            idx (int): Index of the sample.
        Returns:
            he (np.array): HE image.
        """
        return io.imread(self.image_paths[idx])

    def get_semantic_mask(self, idx):
        """Return the semantic mask at the given index.
        Args:
            idx (int): Index of the sample.
        Returns:
            mask (np.array): Semantic mask.
        """
        mask_binary = io.imread(self.mask_paths[idx])
        class_idx = self._get_class_idx(self._get_cell_type(self.mask_paths[idx]))
        mask = ((mask_binary == 1)*class_idx).astype('uint8')
        return mask

    def get_instance_mask(self, idx):
        """The SegPath dataset does not have instance masks."""
        return None

    def get_sample_name(self, idx):
        return None

    def get_sample_names(self):
        return None

    def __repr__(self):
        return None