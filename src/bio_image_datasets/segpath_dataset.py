import os
from skimage import io
from skimage.transform import rescale
from bio_image_datasets.dataset import Dataset



class SegPath(Dataset):
    def __init__(self, local_path, microns_per_pixel=0.5):
        super().__init__(local_path)
        """
        Initialize the SegPath dataset, which consists of HE images and semantic masks.
        for 8 cell types: 
            Epithelium, Lymphocyte, Smooth Muscle, Red Blood Cells, Leukocyte, 
            Endothelium, Plasma Cell, Myeloid Cell.
        for each HE image there is a corresponding semantic mask with one cell type.

        Args:
            local_path (str): Path to the dataset
            microns_per_pixel (float): Microns per pixel in the dataset
        Parameters:
            local_path (str): Path to the dataset
            original_microns_per_pixel (float): Microns per pixel in the original dataset
            microns_per_pixel (float): Microns per pixel in the dataset
            all_cell_types (dict): All cell types in the dataset
        """
        self.local_path = local_path
        self.original_microns_per_pixel = 0.220818
        self.microns_per_pixel = microns_per_pixel
        self.all_cell_types = [
            'panCK_Epithelium', 'CD3CD20_Lymphocyte', 'aSMA_SmoothMuscle', 'CD235a_RBC',
            'CD45RB_Leukocyte', 'ERG_Endothelium', 'MIST1_PlasmaCell', 'MNDA_MyeloidCell']
        self._load_data()

    def _load_data(self):
        """
        Load the data paths for the HE images and the semantic masks;
        the dataset is naturally split into 8 folders, one for each cell type;
        as the dataset is quite large in total this function just loads all the folders 
        existing and adjusts the mapping_dict accordingly.

        Args:
            load_cell_types (list): List of cell type abbreviations to load
            cell_types (dict): Mapping from cell type abbreviations to cell types
        Parameters:
            mapping_dict (dict): Mapping from class indices to cell types
            ext_HE (str): File extension for the HE images
            ext_sem_mask (str): File extension for the semantic masks
            sample_names (list): List of base names of the images
            image_paths (list): List of paths to the HE images
            annotated_class (list): Cell type wich was annotated in the image (one per image) as id
        """
        self.mapping_dict = {0: 'Background'}
        self.ext_HE = '_HE.png'
        self.ext_sem_mask = '_mask.png'
        self.sample_names = []  # list of base name of each image
        self.image_paths = []
        self.annotated_class = []  # stores the annotated cell type for each image as index
        for f in sorted(os.listdir(self.local_path)):
            if f in self.all_cell_types:
                cell_type_idx = len(self.mapping_dict)
                self.mapping_dict[cell_type_idx] = f
                path = os.path.join(self.local_path, f)
                files = sorted(os.listdir(path))
                self.annotated_class += [cell_type_idx] * len(files)
                samples = [g.replace(self.ext_HE, '') for g in files if g.endswith(self.ext_HE)]
                self.sample_names += samples
                self.image_paths += [
                    os.path.join(path, s).replace(self.ext_HE, '') for s in samples]

    def download(self):
        """The SegPath dataset is avaliable via zenodo (see https://dakomura.github.io/SegPath/)"""
        pass

    def _resize(self, image, interp='bilinear'):
        """
        Resize an image to the desired resolution.
        
        Args:
            image (np.array): Image to resize.
            interp (str): Interpolation method options: 'bilinear', 'nearest'.
        Returns:
            np.array: Resized image.
        """
        mode = 0 if interp == 'nearest' else 1
        anti_aliasing = mode == 1
        scale = self.original_microns_per_pixel / self.microns_per_pixel
        channel_axis = 2 if image.ndim == 3 else None
        return rescale(
            image, scale, order=mode, anti_aliasing=anti_aliasing,
            preserve_range=True, channel_axis=channel_axis).astype('uint8')

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the HE image and the semantic mask at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            he (np.array; HxWx3): HE image.
            semantic_mask (np.array; HxW): Semantic mask.
        """
        he = self.get_he(idx)
        semantic_mask = self.get_semantic_mask(idx)
        return he, semantic_mask

    def get_he(self, idx):
        """
        Return the HE image (np.array; HxWx3) at the given index.

        Args:
            idx (int): The index of the image to retrieve.
        Returns:
            np.array: The HE image at the specified index with shape (HxWx3).
        """
        image = io.imread(self.image_paths[idx]+self.ext_HE)
        return self._resize(image)

    def get_semantic_mask(self, idx):
        """
        Return the semantic mask at the given index
        
        Args:
            idx (int): The index of the semantic mask to retrieve.
        Returns:
            np.array: The semantic mask at the specified index with shape (HxW).
        """
        mask_binary = io.imread(self.image_paths[idx]+self.ext_sem_mask)
        mask_binary = self._resize(mask_binary, interp='nearest')
        class_idx = self.annotated_class[idx]
        return ((mask_binary == 1)*class_idx).astype('uint8')
    
    def get_mapping_dict(self):
        """
        Return the mapping dict for the dataset.
        
        Args:
            None
        Returns:
            dict: The mapping dict for the dataset.
        """
        return self.mapping_dict

    def get_instance_mask(self, idx):
        """The SegPath dataset does not have instance masks."""
        return None

    def get_sample_name(self, idx):
        """
        Return the sample name at the given index.
        
        Args:
            idx (int): Index of the sample.
        Returns:
            None
        """
        return self.sample_names[idx]

    def get_sample_names(self):
        """
        Return the list of sample names.
        
        Args:
            None
        Returns:
            list: List of sample names.
        """
        return self.sample_names

    def __repr__(self):
        return None