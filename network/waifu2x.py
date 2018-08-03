from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import json

class waifu2x:
  def __init__(self):
    self.path = './model/waifu2x.json'
    self.params = []
    self.param_id = 0
    with open(self.path, 'rb') as f:
      self.params.append(json.load(f))

    self.lr = 0.0003
    self.in_train = False
    self.input_shape = (self.params[self.param_id][0]['nInputPlane'], 32, 32)

  def network(self):
    network = Sequential()
    network.add(Conv2D(
      self.params[self.param_id][0]['nOutputPlane'],
      (self.params[self.param_id][0]['kH'], self.params[self.param_id][0]['kW']),
      kernel_initializer='zero',
      padding='same',
      #weights=[np.array(params[param_id][0]['weight']), np.array(params[param_id][0]['bias'])],
      use_bias=True,
      input_shape=self.input_shape))

    network.add(LeakyReLU(0.1))

    for param in self.params[self.param_id][1:]:
        network.add(Conv2D(
            param['nOutputPlane'],
            (self.params[self.param_id][0]['kH'], self.params[self.param_id][0]['kW']),
            kernel_initializer='zero',
            padding='same',
            #weights=[np.array(param['weight']), np.array(param['bias'])],
            use_bias=True))
        network.add(LeakyReLU(0.1))

    adam = Adam(lr=self.lr)
    network.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return network
