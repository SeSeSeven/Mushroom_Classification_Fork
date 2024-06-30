import unittest
from src.data.make_dataset import prepare_data
import os

class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        self.dataset_path = "data/Mushroom_Image_Dataset"
        self.output_path = "data"

    def test_prepare_data(self):
        prepare_data(self.dataset_path, self.output_path)
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'train')))
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'val')))
        self.assertTrue(os.path.exists(os.path.join(self.output_path, 'test')))

if __name__ == '__main__':
    unittest.main()
