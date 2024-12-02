import os
from skimage import io
from bio_image_datasets.dataset import Dataset



class SegPath(Dataset):
    def __init__(self, local_path):
        super().__init__(local_path)
        """
        Initialize the SegPath dataset, which consists of HE images and semantic masks.
        for 8 cell types (short): EPI, LYM, MUS, RBC, LEU, END, PLA, MYE; 
        for each HE image there is a corresponding semantic mask with one cell type.

        Args:
            local_path (str): Path to the dataset.
        """
        self.local_path = local_path
        self.load_data()

    def load_data(self):
        """
        Load the data paths for the HE images and the semantic masks.

        Args:
            load_cell_types (list): List of cell type abbreviations to load.
            cell_types (dict): Mapping from cell type abbreviations to cell types.
        Parameters:
            all_cell_types (dict): All cell types in the dataset.
            classes (dict): Mapping from class indices to class names.
            ext_HE (str): File extension for the HE images.
            ext_sem_mask (str): File extension for the semantic masks.
            image_paths (list): List of paths to the HE images.
        """
        self.all_cell_types = [
            'panCK_Epithelium', 'CD3CD20_Lymphocyte', 'aSMA_SmoothMuscle', 'CD235a_RBC',
            'CD45RB_Leukocyte', 'ERG_Endothelium', 'MIST1_PlasmaCell', 'MNDA_MyeloidCell']
        self.mapping_dict = {0: 'Background'}
        self.ext_HE = '_HE.png'
        self.ext_sem_mask = '_mask.png'
        self.image_paths = []
        self.image_cell_types = []
        for f in sorted(os.listdir(self.local_path)):
            if f in self.all_cell_types:
                self.mapping_dict[len(self.mapping_dict)] = f
                path = os.path.join(self.local_path, f)
                files = sorted(os.listdir(path))
                self.image_cell_types += [f]*len(files)
                self.image_paths += [
                    os.path.join(path, g).replace(self.ext_HE, '') 
                    for g in files if g.endswith(self.ext_HE)]
    
    def _get_class_idx(self, cell_type):
        """Get the class index (int) from the cell type (str)"""
        for key, value in self.mapping_dict.items():
            if value == cell_type:
                return key
        return 0

    def download(self):
        """The SegPath dataset is avaliable via zenodo (see https://dakomura.github.io/SegPath/)"""
        pass

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the HE image and the semantic mask at the given index.
        
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
        """Return the HE image (np.array; HxWx3) at the given index (int)"""
        return io.imread(self.image_paths[idx]+self.ext_HE)

    def get_semantic_mask(self, idx):
        """Return the semantic mask (np.array; HxW) at the given index (int)."""
        mask_binary = io.imread(self.image_paths[idx]+self.ext_sem_mask)
        class_idx = self._get_class_idx(self.image_cell_types[idx])
        mask = ((mask_binary == 1)*class_idx).astype('uint8')
        return mask
    
    def get_mapping_dict(self):
        """Return the mapping dict for the dataset."""
        return self.mapping_dict

    def get_instance_mask(self, idx):
        """The SegPath dataset does not have instance masks."""
        return None

    def get_sample_name(self, idx):
        """ The SegPath dataset does not have sample names."""
        return None

    def get_sample_names(self):
        """ The SegPath dataset does not have sample names."""
        return None

    def __repr__(self):
        return None