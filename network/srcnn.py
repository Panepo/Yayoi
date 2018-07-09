from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam

def networkTrain():
  network = Sequential()
  network.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
                   activation='relu', border_mode='valid', bias=True, input_shape=(32, 32, 1)))
  network.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   activation='relu', border_mode='same', bias=True))
  network.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                   activation='linear', border_mode='valid', bias=True))
  adam = Adam(lr=0.0003)
  network.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
  return network
    
def networkInference():
  network = Sequential()
  network.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
                   activation='relu', border_mode='valid', bias=True, input_shape=(None, None, 1)))
  network.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                   activation='relu', border_mode='same', bias=True))
  network.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                   activation='linear', border_mode='valid', bias=True))
  adam = Adam(lr=0.0003)
  network.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
  return network
    
if __name__ == "__main__":
  print("Hello world!")