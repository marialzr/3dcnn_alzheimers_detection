import numpy as np
import pandas as pd

from keras.utils import np_utils

from repository.dataset_type import DatasetType

class Dataset:
    def __init__(self, config):
        self.__config = config
        self.__file_number = 0
        self.__path = '../datasets/img_array_mri/'
        self.__labels_path = '../datasets/'

    def load_images_from_one_file(self, filename):
        self.__file_number = self.__file_number + 1
        dataset = np.load(filename)
        dataset = dataset.astype('float32')
        print('Successfully loaded dataset number %d from 32 sets' % self.__file_number)
        return dataset

    def load_train_set(self):
        path = self.__path+'img_array_test_6k_1.npy'
        final_data = self.load_images_from_one_file(path)
        for i in range(2, 23):
            path = self.__path+'img_array_train_6k_%d.npy'
            loaded_data = self.load_images_from_one_file(path%i)
            final_data = np.vstack((final_data, loaded_data))
        final_data = self.reshape_images(final_data)
        return final_data

    def load_valid_set(self):
        path = self.__path + 'img_array_valid_6k_1.npy'
        final_data = self.load_images_from_one_file(path)
        for i in range(2, 6):
            path = self.__path + 'img_array_valid_6k_%d.npy'
            loaded_data = self.load_images_from_one_file(path%i)
            final_data = np.vstack((final_data, loaded_data))
        final_data = self.reshape_images(final_data)
        return final_data

    def load_test_set(self):
        path = self.__path + 'img_array_test_6k_1.npy'
        final_data = self.load_images_from_one_file(path)
        for i in range(2, 6):
            path = self.__path + 'img_array_test_6k_%d.npy'
            loaded_data = self.load_images_from_one_file(path % i)
            final_data = np.vstack((final_data, loaded_data))
        final_data = self.reshape_images(final_data)
        return final_data

    def scale_images(self, dataset):
        for i in range(len(dataset)):
            max_val = np.amax(dataset[i][0])
            max_val = max_val.astype('float32')
            dataset[i][0] = dataset[i][0].astype('float32')
            dataset[i][0] /= max_val
        print('Images scaled')
        return dataset

    def load_images_by_type(self, type):
        if DatasetType.TRAIN == type:
            return self.load_train_set()
        elif DatasetType.VALID == type:
            return self.load_valid_set()
        elif DatasetType.TEST == type:
            return self.load_test_set()

    def load_images(self):
        train = self.load_images_by_type(DatasetType.TRAIN)
        print(train.shape)
        train_images = self.scale_images(train)
        valid = self.load_images_by_type(DatasetType.VALID)
        valid_images = self.scale_images(valid)
        test = self.load_images_by_type(DatasetType.TEST)
        test_images = self.scale_images(test)
        return train_images, valid_images, test_images

    def load_labels_by_type(self, filename, datasetType):
        print("!!!!")
        print(filename)
        labels = pd.read_csv(filename)
        labels = labels.loc[labels['type'] == datasetType]
        labels = np.asarray(labels.group)
        labels = labels.reshape((-1, 1))
        labels = labels.astype('float32')
        labels = np.subtract(labels, 1, dtype='float32')
        labels = np_utils.to_categorical(labels, int(self.__config["no_classes"]), dtype='float32')
        return labels

    def load_labels(self):
        train_labels = self.load_labels_by_type(self.__labels_path+'adni_data_labels.csv',
                                           DatasetType.TRAIN.value)
        val_labels = self.load_labels_by_type(self.__labels_path+ 'adni_data_labels.csv',
                                         DatasetType.VALID.value)
        test_labels = self.load_labels_by_type(self.__labels_path+'adni_data_labels.csv',
                                          DatasetType.TEST.value)
        print('Successfully loaded labels.')
        return train_labels, val_labels, test_labels

    def reshape_images(self, dataset):
        # Reshape the loaded dataset to the appropriate format.
        dataset = np.expand_dims(dataset, axis=1)
        dataset = np.reshape(dataset, (-1, 1, int(self.__config["channels"]), int(self.__config["width"]), int(self.__config["height"])))
        print('Successfully reshaped')
        return dataset