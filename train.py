# %%
import numpy as np 
from processData import load_data
from MyNN import * 

np.random.seed(90)

epochs = 15
batch_size = 300
decay_rate = 0.5
lr, hidden1_dim, hidden2_dim, weight_decay=0.1, 256, 256, 0.001

X_train, Y_train, X_valid, Y_valid, X_test, Y_test, classes_num, input_dim = load_data()

model = MyModel(input_dim, hidden1_dim, hidden2_dim, classes_num)
model.optimize_visualize(X_train, Y_train, X_valid, Y_valid, epochs, batch_size, lr, decay_rate, weight_decay, print_log=True)

save_model(model)

# test evaluation 
# precision = model.precision(X_test, Y_test)
# print('Precision of test set: ', precision)