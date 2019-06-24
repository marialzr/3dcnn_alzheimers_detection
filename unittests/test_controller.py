import unittest
import os
import json
from service.controller import Controller


class TestControllerMethods(unittest.TestCase):
    def setUp(self):
        data = {}
        data['no_classes'] = "3"
        data['no_epochs'] = "20"
        data['batch_size'] = "10"
        data['class_names'] = "CN MCI AD"
        data['channels'] = "62"
        data['width'] = "96"
        data['height'] = "96"
        data['optimizer'] = "adam"
        data['loss'] = "categorical_crossentropy"
        data['download_imgs'] = "../repository"
        data['weights_path'] = "../repository/data.h5"

        with open('cfg_test.json', 'w+') as outfile:
            json.dump(data, outfile)

        self.file_path = 'cfg_test.json'
        self.controller = Controller(self.file_path)

    def tearDown(self) -> None:
        os.remove("cfg_test.json")
        print("File Removed!")

    def test_parse_config_file(self):
        data = self.controller.parse_config_file(self.file_path)
        self.assertEqual(data["no_classes"], "3")
        self.assertEqual(data["no_epochs"], "20")
        self.assertEqual(data["batch_size"], "10")
        self.assertEqual(data["class_names"], "CN MCI AD")
        self.assertEqual(data["channels"], "62")
        self.assertEqual(data["width"], "96")
        self.assertEqual(data["height"], "96")
        self.assertEqual(data["optimizer"], "adam")
        self.assertEqual(data["loss"], "categorical_crossentropy")
        self.assertEqual(data["download_imgs"], "../repository")
        self.assertEqual(data["weights_path"], "../repository/data.h5")
