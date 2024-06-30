import unittest
from src.models.model import get_model

class TestModel(unittest.TestCase):

    def test_model_creation(self):
        model = get_model(num_classes=9)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
