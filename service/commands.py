from service.controller import Controller

class Commands:
    def __init__(self, new_model_file=None):
        self.__controller = Controller(r'C:\Users\camel\PycharmProjects\alzheimers_detection - Copy\utils\config.json', new_model_file)

    def predict(self, path_to_image):
         return self.__controller.predict(path_to_image)


    def plot(self, img_path, filename, store_path_dir):
        self.__controller.plot(img_path, filename, store_path_dir)
        return

    def train(self):
        print("start training...")
        return self.__controller.train()