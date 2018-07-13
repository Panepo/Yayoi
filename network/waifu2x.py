from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

class waifu2x:
  def __init__(self):
    self.layers = 7
    self.nOutputPlane = [32, 32, 64, 64, 128, 128, 1]
    self.kH = [3, 3, 3, 3, 3, 3, 3]
    self.kW = [3, 3, 3, 3, 3, 3, 3]
    self.inputShape = (32, 32, 1)
    self.lr = 0.0003
    self.in_train = False

  def network(self):
    network = Sequential()

    if (self.in_train):
      network.add(Conv2D(
        self.nOutputPlane[0], 
        self.kH[0],
        self.kW[0],
        kernel_initializer='zero',
        padding='same', 
        use_bias=True, 
        input_shape=self.inputShape))
    else:
      network.add(Conv2D(
        self.nOutputPlane[0], 
        self.kH[0],
        self.kW[0],
        kernel_initializer='zero',
        padding='same', 
        use_bias=True, 
        input_shape=self.inputShape))
    
    network.add(LeakyReLU(0.1))
    for i in range(1, self.layers-1):
      network.add(Conv2D(
        self.nOutputPlane[i], 
        self.kH[i],
        self.kW[i],
        kernel_initializer='zero',
        padding='same', 
        use_bias=True))
      network.add(LeakyReLU(0.1))

    return network