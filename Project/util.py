import numpy as np
import h5py

NUM_DATA_FILE = 9
TEST_SIZE = 0
TRAIN_SIZE = 288 - TEST_SIZE


def load_data():
    train_x = np.zeros([NUM_DATA_FILE * TRAIN_SIZE, 22, 1000,1])
    train_y = np.zeros([NUM_DATA_FILE * TRAIN_SIZE])
    test_x = np.zeros([NUM_DATA_FILE * TEST_SIZE,22,1000,1])
    test_y = np.zeros([NUM_DATA_FILE * TEST_SIZE])
    for i in range(NUM_DATA_FILE):
        A = h5py.File("./data/A0%dT_slice.mat"%(i+1), "r")
        x = np.copy(A['image'])
        x = x.reshape((288,25,1000,1))
        x = x[:,:22,:,:]
        y = np.copy(A['type'])[0,:288]
        y[y == 769] = 0
        y[y == 770] = 1
        y[y == 771] = 2
        y[y == 772] = 3
        
        print (np.unique(np.where(np.isnan(x))[0]))

        idx = np.random.choice(288,TEST_SIZE,replace=False)

        test_x[i*TEST_SIZE:(i+1)*TEST_SIZE,:,:,:] = x[idx,:,:,:]
        test_y[i*TEST_SIZE:(i+1)*TEST_SIZE] = y[idx]

        _x,_y = getUnselectedData(x,y,idx)

        train_x[i*TRAIN_SIZE:(i+1)*TRAIN_SIZE,:,:,:] = _x
        train_y[i*TRAIN_SIZE:(i+1)*TRAIN_SIZE] = _y
    
    train_x,train_y = clear_nan(train_x,train_y)
    test_x,test_y = clear_nan(test_x, test_y)
 

        
    return (train_x,train_y,test_x,test_y)

def getUnselectedData(x,y,idx):
    _x = np.zeros([TRAIN_SIZE,22,1000,1])
    _y = np.zeros([TRAIN_SIZE])
    counter = 0
    for i in range(288):
        if i not in idx:
            _x[counter,:,:,:] = x[i,:,:,:]
            _y[counter] = y[counter]
            counter += 1
    return _x,_y

def clear_nan(x,y):
    tmp = np.where(np.isnan(x))
    nan_idx = np.unique(tmp[0])
    print (nan_idx)
    _x = np.zeros((x.shape[0] - len(nan_idx),x.shape[1],x.shape[2],x.shape[3]))
    _y = np.zeros(x.shape[0] - len(nan_idx))
    counter = 0
    for i in range(x.shape[0]):
        if i not in nan_idx:
            _x[counter,:,:,:] = x[i,:,:,:]
            _y[counter] = y[i]
            counter += 1
    return _x,_y
                




