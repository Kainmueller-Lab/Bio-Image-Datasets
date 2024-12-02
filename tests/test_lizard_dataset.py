import unittest
import numpy as np
import os
import tempfile
import shutil
from bio_image_datasets.lizard_dataset import LizardDataset
import scipy.io as sio
from PIL import Image

class TestLizardDataset(unittest.TestCase):
    """Test suite for the LizardDataset class."""

    @classmethod
    def setUpClass(cls):
        """Set up the dataset with mock data for all tests."""
        # Create a temporary directory
        cls.temp_dir = tempfile.mkdtemp()

        # Set up mock data directories
        cls.mock_local_path = cls.temp_dir
        image_dirs = [
            os.path.join(cls.mock_local_path, 'lizard_images1', 'Lizard_Images1'),
            os.path.join(cls.mock_local_path, 'lizard_images2', 'Lizard_Images2')
        ]
        label_dir = os.path.join(cls.mock_local_path, 'lizard_labels', 'Lizard_Labels', 'Labels')

        # Create directories
        for dir_path in image_dirs + [label_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Create mock image and label files
        cls.sample_filenames = ['sample1.png', 'sample2.png']
        for image_dir in image_dirs:
            for filename in cls.sample_filenames:
                # Create a mock image file
                image_path = os.path.join(image_dir, filename)
                mock_image = Image.new('RGB', (100, 100), color=(255, 0, 0))
                mock_image.save(image_path)

                # Create a corresponding mock .mat label file
                label_name = filename.replace('.png', '.mat')
                label_path = os.path.join(label_dir, label_name)
                mock_label_data = {
                    'inst_map': np.ones((100, 100), dtype=np.int32),
                    'class': np.array([1]),      # Changed here
                    'id': np.array([1]),         # Changed here
                    'bbox': np.array([[0, 99, 0, 99]]),
                    'centroid': np.array([[50, 50]])
                }
                sio.savemat(label_path, mock_label_data)

        # Instantiate the dataset with the mock data
        cls.dataset = LizardDataset(local_path=cls.mock_local_path)
        cls.sample_idx = 0  # Index of the sample to test
        cls.total_samples = len(cls.dataset)


    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after tests."""
        shutil.rmtree(cls.temp_dir)

    def test_len(self):
        """Test the __len__ method."""
        self.assertTrue(self.total_samples > 0, "Dataset should contain at least one sample.")

    def test_get_he(self):
        """Test the get_he method."""
        image = self.dataset.get_he(self.sample_idx)
        self.assertIsInstance(image, np.ndarray, "Image should be a NumPy array.")
        self.assertEqual(image.ndim, 3, "Image should have 3 dimensions (C, H, W) or (H, W, C).")
        self.assertEqual(image.shape[0], 3, "Image should have 3 channels (RGB).")
        self.assertTrue(np.max(image) <= 255 and np.min(image) >= 0, "Image pixel values should be between 0 and 255.")

    def test_get_semantic_mask(self):
        """Test the get_semantic_mask method."""
        semantic_mask = self.dataset.get_semantic_mask(self.sample_idx)
        self.assertIsInstance(semantic_mask, np.ndarray, "Semantic mask should be a NumPy array.")
        self.assertEqual(semantic_mask.ndim, 2, "Semantic mask should have 2 dimensions (H, W).")
        unique_classes = np.unique(semantic_mask)
        self.assertTrue(len(unique_classes) >= 1, "Semantic mask should contain at least one class.")
        self.assertTrue(np.all(unique_classes >= 0), "Class labels should be non-negative integers.")

    def test_get_instance_mask(self):
        """Test the get_instance_mask method."""
        instance_mask = self.dataset.get_instance_mask(self.sample_idx)
        self.assertIsInstance(instance_mask, np.ndarray, "Instance mask should be a NumPy array.")
        self.assertEqual(instance_mask.ndim, 2, "Instance mask should have 2 dimensions (H, W).")
        unique_instances = np.unique(instance_mask)
        self.assertTrue(len(unique_instances) >= 1, "Instance mask should contain at least one instance.")
        self.assertTrue(np.all(unique_instances >= 0), "Instance IDs should be non-negative integers.")

    def test_get_sample_name(self):
        """Test the get_sample_name method."""
        sample_name = self.dataset.get_sample_name(self.sample_idx)
        self.assertIsInstance(sample_name, str, "Sample name should be a string.")
        self.assertTrue(len(sample_name) > 0, "Sample name should not be empty.")

    def test_get_sample_names(self):
        """Test the get_sample_names method."""
        sample_names = self.dataset.get_sample_names()
        self.assertIsInstance(sample_names, list, "Sample names should be a list.")
        self.assertEqual(len(sample_names), self.total_samples, "Number of sample names should match dataset length.")
        self.assertTrue(all(isinstance(name, str) for name in sample_names), "All sample names should be strings.")

    def test_get_item(self):
        """Test the __getitem__ method."""
        sample = self.dataset[self.sample_idx]
        self.assertIsInstance(sample, dict, "Sample should be a dictionary.")
        expected_keys = {'image', 'semantic_mask', 'instance_mask', 'sample_name'}
        self.assertTrue(expected_keys.issubset(sample.keys()), "Sample dictionary should contain all expected keys.")
        # Test image
        image = sample['image']
        self.assertIsInstance(image, np.ndarray, "Image should be a NumPy array.")
        self.assertEqual(image.ndim, 3, "Image should have 3 dimensions (C, H, W) or (H, W, C).")
        self.assertEqual(image.shape[0], 3, "Image should have 3 channels (RGB).")
        # Test semantic mask
        semantic_mask = sample['semantic_mask']
        self.assertIsInstance(semantic_mask, np.ndarray, "Semantic mask should be a NumPy array.")
        # Test instance mask
        instance_mask = sample['instance_mask']
        self.assertIsInstance(instance_mask, np.ndarray, "Instance mask should be a NumPy array.")
        # Test sample name
        sample_name = sample['sample_name']
        self.assertIsInstance(sample_name, str, "Sample name should be a string.")

    def test_repr(self):
        """Test the __repr__ method."""
        representation = repr(self.dataset)
        self.assertIsInstance(representation, str, "__repr__ should return a string.")
        self.assertIn("LizardDataset", representation, "__repr__ should include the class name.")

    def test_load_label(self):
        """Test the _load_label helper method."""
        label_data = self.dataset._load_label(self.sample_idx)
        self.assertIsInstance(label_data, dict, "Label data should be a dictionary.")
        expected_keys = {'inst_map', 'class', 'id', 'bbox', 'centroid'}
        self.assertTrue(expected_keys.issubset(label_data.keys()), "Label data should contain all expected keys.")
        # Test inst_map
        inst_map = label_data['inst_map']
        self.assertIsInstance(inst_map, np.ndarray, "inst_map should be a NumPy array.")
        self.assertEqual(inst_map.ndim, 2, "inst_map should have 2 dimensions (H, W).")
        # Test classes
        classes = label_data['class']
        self.assertIsInstance(classes, np.ndarray, "classes should be a NumPy array.")
        # Test nuclei_id
        nuclei_id = label_data['id']
        self.assertIsInstance(nuclei_id, np.ndarray, "nuclei_id should be a NumPy array.")

if __name__ == '__main__':
    unittest.main()
