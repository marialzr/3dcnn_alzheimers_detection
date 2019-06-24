import json

from model.base_convnet import BaseConvNet
from model.convnet_v1 import ConvNetV1
from model.convnet_v2 import ConvNetV2
from model.convnet_v3 import ConvNetV3
from model.convnet_v4 import ConvNetV4
from model.custom_convnet import CustomNet


class Controller:
    def __init__(self, config_file, train_version=None, new_model_file=None):
        self.create_model(train_version, config_file, new_model_file)

    def create_model(self, train_version, config_file, new_model_file):
        if new_model_file is None:
            self.__model = BaseConvNet(self.parse_config_file(config_file))
        elif train_version == 1:
            self.__model = ConvNetV1(self.parse_config_file(config_file))
        elif train_version == 2:
            self.__model = ConvNetV2(self.parse_config_file(config_file))
        elif train_version == 3:
            self.__model = ConvNetV3(self.parse_config_file(config_file))
        elif train_version == 4:
            self.__model = ConvNetV4(self.parse_config_file(config_file))
        else:
            architecture = self.parse_config_file(new_model_file)
            self.__model = CustomNet(self.parse_config_file(config_file), architecture)
            print("architecture!")
            print(architecture)

    def predict(self, path_to_image):
        prediction = self.__model.evaluate_image(path_to_image)

        if prediction[0] == 1.0:
            return 'CN'
        elif prediction[1] == 1.0:
            return 'MCI'
        elif prediction[2] == 1.0:
            return 'AD'

    def plot(self, img_path, filename, store_path_dir):
        self.__model.plot_image(img_path, filename, store_path_dir)
        return

    def train(self):
        print("start training...")
        return self.__model.train()

    @staticmethod
    def parse_config_file(config_file='C:/Users/camel/PycharmProjects/alzheimers_detection - Copy/utils/config.json'):
        with open(config_file) as f:
            data = json.load(f)
        print(data)
        return data
