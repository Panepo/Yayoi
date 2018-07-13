from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

class waifu2x:
  def __init__(self):
    self.layers = 7
    self.filters = [32, 32, 64, 64, 128, 128, 1]
    self.kernelSize = [(3,3), (3,3), (3,3), (3,3), (3,3), (3,3), (3,3)]
    self.inputShape = (32, 32, 1)
    self.lr = 0.0003
    self.in_train = False

  def network(self):
    network = Sequential()

    if (self.in_train):
      network.add(Conv2D(
        filters = self.filters[0], 
        kernel_size = self.kernelSize[0],
        kernel_initializer='zero',
        padding='valid', 
        use_bias=True, 
        input_shape=self.inputShape))
    else:
      network.add(Conv2D(
        filters = self.filters[0], 
        kernel_size = self.kernelSize[0],
        kernel_initializer='zero',
        padding='valid', 
        use_bias=True, 
        input_shape=(None, None, 1)))
    
    network.add(LeakyReLU(0.1))
    for i in range(1, self.layers-1):
      network.add(Conv2D(
        filters = self.filters[i], 
        kernel_size = self.kernelSize[i],
        kernel_initializer='zero',
        padding='same', 
        use_bias=True))
      network.add(LeakyReLU(0.1))

    network.add(Conv2D(filters=self.filters[self.layers-1], kernel_size = self.kernelSize[self.layers-1], kernel_initializer='zero',
                       padding='valid', use_bias=True))
    adam = Adam(lr=self.lr)
    network.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return network