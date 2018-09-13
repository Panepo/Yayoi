from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.optimizers import Adam

class srcnn:
  def __init__(self):
    self.layers = 3
    self.filter = [64, 32, 1]
    self.kernelSize = [(9,9), (3,3), (5,5)]
    self.inputShape = (32, 32, 1)
    self.lr = 0.0003
    self.in_train = False
    self.dataPath = ""

  def network(self):
    network = Sequential()

    if (self.in_train):
      network.add(Conv2D(filters=self.filter[0], kernel_size = self.kernelSize[0], kernel_initializer='glorot_uniform',
                         activation='relu', padding='valid', use_bias=True, input_shape=self.inputShape))
    else:
      network.add(Conv2D(filters=self.filter[0], kernel_size = self.kernelSize[0], kernel_initializer='glorot_uniform',
                         activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))

    for i in range(1, self.layers-1):
      network.add(Conv2D(filters=self.filter[i], kernel_size = self.kernelSize[i], kernel_initializer='glorot_uniform',
                         activation='relu', padding='same', use_bias=True))
      #network.add(BatchNormalization())

    network.add(Conv2D(filters=self.filter[self.layers-1], kernel_size = self.kernelSize[self.layers-1], kernel_initializer='glorot_uniform',
                       activation='linear', padding='valid', use_bias=True))
    adam = Adam(lr=self.lr)
    network.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return network
