import unittest
import os
from src.visualization.visualize import visualize_predictions

class TestVisualize(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'data'
        self.output_dir = 'reports/figures'
        self.predictions = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Dummy predictions for testing

    def test_visualize_predictions(self):
        visualize_predictions(self.data_dir, self.predictions, output_dir=self.output_dir, num_images=9)
        output_path = os.path.join(self.output_dir, 'predictions_visualization.png')
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()
