from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


class Turtle:
    @staticmethod
    def OneConvLayer(input_shape):
        """
        function for 1 conv layer model
        :param input_shape: shape of the image
        :return: model structure
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(9, activation='softmax'))
        return model

    @staticmethod
    def TwoConvLayer(input_shape):
        """
        function for 2 conv layer model
        :param input_shape: shape of the image
        :return: model structure
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(9, activation='softmax'))
        return model

    @staticmethod
    def ThreeConvLayer(input_shape):
        """
        function for 3 conv layer model
        :param input_shape: shape of the image
        :return: model structure
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(0.4))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(9, activation='softmax'))
        return model

    @staticmethod
    def FourConvLayer(input_shape):
        """
        function for 4 conv layer model
        :param input_shape: shape of the image
        :return: model structure
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(9, activation='softmax'))
        return model

    def getModel(self, modelID, input_shape):
        """

        :param modelID: ID of the model to return
        :param input_shape: shape of the img
        :return: model
        """
        if modelID == 1:
            return self.OneConvLayer(input_shape)
        elif modelID == 2:
            return self.TwoConvLayer(input_shape)
        elif modelID == 3:
            return self.ThreeConvLayer(input_shape)
        elif modelID == 4:
            return self.FourConvLayer(input_shape)
