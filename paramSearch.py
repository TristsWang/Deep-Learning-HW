import numpy as np 
from MyNN import MyModel 
from itertools import product
from processData import load_data 
from tqdm import tqdm 

np.random.seed(1234)

X_train, Y_train, X_valid, Y_valid, X_test, Y_test, classes_num, input_dim = load_data()

epochs = 10
batch_size = 250 
lr_decay_rate = 0.5

lrs = [1e-1, 1e-2]
hidden1_dims = [64, 128, 256]
hidden2_dims = [64, 128, 256]
weight_decays = [1, 1e-1, 1e-3]

possible_params = dict(lrs=lrs, hidden1_dims=hidden1_dims, hidden2_dims=hidden2_dims, weight_decays=weight_decays)

def search_param(possible_params, log_file_name):
    result = dict()

    f = open(log_file_name, 'w')
    f.write('lr,hidden1_dim,hidden2_dim,weight_decay,training_loss,training_acc,validation_loss,validation_acc\n')

    params = product(*possible_params.values())
    round = 0
    for param in tqdm(params):
        lr, hidden1_dim, hidden2_dim, weight_decay = param 
        print()
        print("Param{}: lr={}, hidden1_dim={}, hidden2_dim={}, weight_decay={}".format(round+1, lr, hidden1_dim, hidden2_dim, weight_decay))
        print()
        
        model = MyModel(input_dim, hidden1_dim, hidden2_dim, classes_num)
        model.optimize_train(X_train, Y_train, epochs, batch_size, lr, lr_decay_rate, weight_decay, print_log=True)
        round += 1
        # use validation accuracy to choose the best combination of parameters 
        acc_valid, loss_valid = model.evaluate(X_valid, Y_valid)
        result[param] = acc_valid
        acc_train, loss_train = model.evaluate(X_train, Y_train)
        
        log = str(lr) + ',' + str(hidden1_dim) + ',' + str(hidden2_dim) + ',' + str(weight_decay) + ',' + str(loss_train) + ',' + str(acc_train) + ',' + str(loss_valid) + ',' + str(acc_valid) + '\n'
        f.write(log)
    
    best_param = max(result, key=result.get)
    print(best_param)
    final_log = 'best parameters: lr={}, hidden1_dim={}, hidden2_dim={}, weight_decay={}'.format(*best_param)
    f.write(final_log)
    print(final_log)
    f.close()

search_param(possible_params, 'search_parameters_2.csv')