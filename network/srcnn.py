from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.optimizers import Adam

class srcnn:
  def __init__(self):
    self.layers = 3
    self.filter = [64, 32, 1]
    self.conv = [9, 3, 5]
    self.inputShape = (32, 32, 1)
    self.lr = 0.0003
    self.in_train = False

  def network(self):
    network = Sequential()

    if (self.in_train):
      network.add(Conv2D(filters=self.filter[0], nb_row=self.conv[0], nb_col=self.conv[0], init='glorot_uniform',
                         activation='relu', padding='valid', bias=True, input_shape=self.inputShape))
    else:
      network.add(Conv2D(filters=self.filter[0], nb_row=self.conv[0], nb_col=self.conv[0], init='glorot_uniform',
                         activation='relu', padding='valid', bias=True, input_shape=(None, None, 1)))
    
    for i in range(1, self.layers-1):
      network.add(Conv2D(filters=self.filter[i], nb_row=self.conv[i], nb_col=self.conv[i], init='glorot_uniform',
                         activation='relu', padding='same', bias=True))
      network.add(BatchNormalization())

    network.add(Conv2D(filters=self.filter[self.layers-1], nb_row=self.conv[self.layers-1], nb_col=self.conv[self.layers-1], init='glorot_uniform',
                       activation='linear', padding='valid', bias=True))
    adam = Adam(lr=self.lr)
    network.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return network