from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Convolution3D, MaxPooling3D, Flatten, Dense, Dropout
from model.base_convnet import BaseConvNet


class ConvNetV2(BaseConvNet):
    def train(self):
        dataset = self.dataset
        train_images, val_images, test_images = dataset.load_images()
        train_labels, val_labels, test_labels = dataset.load_labels()
        model = self.build_model()
        scheduler = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=1e-5)
        self.train_model(model, train_images, train_labels, val_images, val_labels, self.optimizer, scheduler)

    def build_model(self):
        model = Sequential()
        model.add(Convolution3D(8, (3, 3, 3), input_shape=self.__input_dim))
        model.add(MaxPooling3D())

        model.add(Convolution3D(8, (3, 3, 3)))
        model.add(MaxPooling3D())

        model.add(Convolution3D(8, (3, 3, 3)))
        model.add(MaxPooling3D())

        model.add(Flatten())
        model.add(Dense(1024, activation='relu', name='dense1'))
        model.add(Dropout(0.5, name='dropout1'))

        model.add(Dense(512, activation='relu', name='dense2'))
        model.add(Dropout(0.5, name='dropout2'))

        model.add(Dense(self.__no_classes, activation='softmax'))

        if self.weights_path:
            model.load_weights(self.weights_path)
        return model