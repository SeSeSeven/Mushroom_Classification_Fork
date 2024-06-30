import unittest
from src.train_model import train_model

class TestTrainModel(unittest.TestCase):

    def test_train_model(self):
        data_dir = "data"
        model_name = "hf_hub:timm/resnet50.a1_in1k"
        num_classes = 9
        train_model(data_dir, model_name, num_classes, epochs=1, batch_size=2, lr=0.001)
        self.assertTrue(True)  # Add actual checks for model training

if __name__ == '__main__':
    unittest.main()
