import h5py
import numpy as np

def h5DataRead(file):
  with h5py.File(file, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    train_data = np.transpose(data, (0, 2, 3, 1))
    train_label = np.transpose(label, (0, 2, 3, 1))
    return train_data, train_label
    
def h5DataWrite(data, labels, output_filename):
  x = data.astype(np.float32)
  y = labels.astype(np.float32)

  with h5py.File(output_filename, 'w') as h:
    h.create_dataset('data', data=x, shape=x.shape)
    h.create_dataset('label', data=y, shape=y.shape)