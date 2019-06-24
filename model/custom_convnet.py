from model.base_convnet import BaseConvNet
from keras.models import Sequential
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.layers import Convolution3D, MaxPooling3D, Flatten, Dense, Dropout


class CustomNet(BaseConvNet):
    pass

    def train(self):
        train_images, val_images, test_images = self.dataset.load_images()
        train_labels, val_labels, test_labels = self.dataset.load_labels()
        model = self.build_model()
        scheduler = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=1e-5)
        self.train_model(model, train_images, train_labels, val_images, val_labels, self.optimizer, scheduler)

    def add_convolutional_layer(self, layer_desc, model, is_first):
        kernel_size = layer_desc["kernel_size"]
        no_filters = layer_desc["no_filters"]
        if is_first:
            model.add(Convolution3D(no_filters, (kernel_size, kernel_size, kernel_size), input_shape=self.input_dim))
        else:
            model.add(Convolution3D(no_filters, (kernel_size, kernel_size, kernel_size)))

    @staticmethod
    def add_pooling_layer(layer_desc, model):
        model.add(MaxPooling3D(pool_size=layer_desc["pool_size"]))

    @staticmethod
    def add_flatten_layer(model):
        model.add(Flatten())

    @staticmethod
    def add_dense_layer(layer_desc, model):
        model.add(Dense(units=layer_desc["no_units"], activation=layer_desc["activation_function"]))

    @staticmethod
    def add_dropout_layer(layer_desc, model):
        model.add(Dropout(layer_desc["rate"]))

    def build_model(self):
        model = Sequential()

        layers = self.architecture["layers"]
        if layers[0]["type"] == "convolution":
            self.add_convolutional_layer(layers[0], model, True)

        for i in range(1, len(layers)):
            layer = layers[i]
            if layer["type"] == "convolution":
                self.add_convolutional_layer(layer, model, False)
            elif layer["type"] == "pooling":
                self.add_pooling_layer(layer, model)
            elif layer["type"] == "flatten":
                self.add_flatten_layer(model)
            elif layer["type"] == "dense":
                self.add_dense_layer(layer, model)
            elif layer["type"] == "dropout":
                self.add_dropout_layer(layer, model)
        return model