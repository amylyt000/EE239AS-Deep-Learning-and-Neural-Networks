from util import *
from cnn import *
import numpy as np
from scnn import *
train_x,train_y,test_x,test_y = load_data()

print("train_x shape",train_x.shape)
print("train_y shape",train_y.shape)

print("test_x shape",test_x.shape)
print("test_y shape",test_y.shape)

print ('num of nan in train_x',np.sum(np.isnan(train_x)))
print ('num of nan in test_x',np.sum(np.isnan(test_x)))

cnn_model = get_cnn_model()
scnn_model = get_scnn_model()
nn_model = get_nn_model()
model = train_cnn(cnn_model,train_x,train_y)



