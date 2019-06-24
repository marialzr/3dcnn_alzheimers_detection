import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from matplotlib import pyplot

from repository.dataset import Dataset

from keras.layers import Convolution3D, MaxPooling3D, Dense, Dropout, Flatten
from keras import backend as K

K.set_image_dim_ordering('th')


class BaseConvNet:
    def __init__(self, params, architecture=None):
        self.no_classes = int(params['no_classes'])
        self.no_epochs = int(params['no_epochs'])
        self.batch_size = int(params['batch_size'])
        self.weights_path = params['weights_path']
        self.architecture = architecture
        self.class_names = params['class_names'].split(" ")
        self.input_dim = (1, int(params['channels']), int(params['width']), int(params['height']))
        self.optimizer = params['optimizer']
        self.loss = params['loss']
        self.tensorboard = TensorBoard(batch_size=self.batch_size)
        print("params:", params)
        self.dataset = Dataset(params)

    def train(self):
        train_images, val_images, test_images = self.dataset.load_images()
        train_labels, val_labels, test_labels = self.dataset.load_labels()
        model = self.build_model()
        scheduler = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=1e-5)
        return self.train_model(model, train_images, train_labels, val_images, val_labels, self.optimizer, scheduler)

    def train_model(self, model, train_images, train_labels, val_images, val_labels, optimizer, scheduler,
                    file_weights='new_model_weights_file.h5'):
        file_model = 'new_model_file.h5'
        model.compile(optimizer=optimizer, loss=self.loss, metrics=["accuracy"])
        model.fit(train_images, train_labels, batch_size=self.batch_size, nb_epoch=self.no_epochs, verbose=1,
                  shuffle=True, validation_data=(val_images, val_labels),
                  callbacks=[scheduler, self.tensorboard])

        self.store_weights(file_model, file_weights)
        return "model trained"

    # evaluate model on 1 image
    def evaluate_image(self, path_to_image):
        my_model = self.build_model()
        my_model.load_weights(self.weights_path)
        my_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])
        img14 = np.load(path_to_image)
        image = np.expand_dims(img14, axis=1)
        prediction = my_model.predict(image)
        prediction_labels = np_utils.to_categorical(np.argmax(prediction, axis=1), self.no_classes)
        return prediction_labels[0]

    # evaluate on test set
    def evaluate_model(self, test_images, test_labels, file_weights='weights_file.h5'):
        loaded_model = self.build_model()
        loaded_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])
        self.evaluate_test_set(test_images, test_labels, loaded_model, file_weights)

    def build_model(self):
        model = Sequential()
        print(self.input_dim)
        print(self.no_classes)
        model.add(Convolution3D(8, (3, 3, 3), input_shape=self.input_dim))
        model.add(MaxPooling3D())

        model.add(Convolution3D(8, (3, 3, 3)))
        model.add(MaxPooling3D())

        model.add(Convolution3D(8, (3, 3, 3)))
        model.add(MaxPooling3D())

        model.add(Flatten())
        model.add(Dense(1000, activation='relu', name='dense1'))
        model.add(Dropout(0.5, name='dropout1'))

        model.add(Dense(500, activation='relu', name='dense2'))
        model.add(Dropout(0.5, name='dropout2'))

        model.add(Dense(self.no_classes, activation='softmax'))

        if self.weights_path:
            model.load_weights(self.weights_path)
        return model

    def plot_image(self, img_path, filename, store_path_dir):
        jet_cmap = plt.cm.jet
        img = np.load(img_path)[0]
        filename = filename.split('.')[0]
        print(filename)

        store_path1 = store_path_dir + "/" + filename + "1.png"
        print(store_path1)

        store_path2 = store_path_dir + "/" + filename + "2.png"
        print(store_path1)
        print(store_path2)

        plot_x = 8
        plot_y = 8

        plot_range = plot_x * plot_y

        fig, axs = plt.subplots(plot_x, plot_y, figsize=(10, 10))
        fig.subplots_adjust(hspace=.5, wspace=.001)

        for ax, d in zip(axs.ravel(), range(plot_range)): ax.axis('off')

        i = 0
        for ax, d in zip(axs.ravel(), img):
            i = i + 1
            ax.imshow(d, cmap=jet_cmap, alpha=.9, interpolation='nearest')
            ax.set_title(str(i))

        print("store path=")
        plt.savefig(store_path1)
        plt.show()
        plt.imshow(img[31], cmap=jet_cmap, alpha=.9, interpolation='bilinear')
        plt.savefig(store_path2)

    def store_weights(self, file_model, file_weights):
        self.model.save(file_model)
        self.model.save_weights(file_weights)
        print("weights stored")

    def evaluate_test_set(self, test_images, test_labels, my_model, weights):
        plt.close('all')
        my_model.load_weights(weights)

        prediction = my_model.predict(test_images)
        prediction_labels = np_utils.to_categorical(np.argmax(prediction, axis=1), self.no_classes)

        print('Accuracy on test data:', accuracy_score(test_labels, prediction_labels))
        print(classification_report(test_labels, prediction_labels, target_names=self.class_names))

        # confusion matrix
        cnf_matrix = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(prediction, axis=1))
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=self.class_names, normalize=False, title='Confusion matrix')

        plt.show()

    @staticmethod
    def show_accuracy(my_model):
        pyplot.title('Accuracy')
        pyplot.plot(my_model.history['acc'], label='train')
        pyplot.plot(my_model.history['val_acc'], label='validation')
        pyplot.ylabel('value')
        pyplot.xlabel('epoch')
        pyplot.legend()
        pyplot.show()

    @staticmethod
    def show_loss(my_model):
        pyplot.title('Loss')
        pyplot.plot(my_model.history['loss'], label='train')
        pyplot.plot(my_model.history['val_loss'], label='validation')
        pyplot.ylabel('value')
        pyplot.xlabel('epoch')
        pyplot.legend()
        pyplot.show()

    @staticmethod
    def plot_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
