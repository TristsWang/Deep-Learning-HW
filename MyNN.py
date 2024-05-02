import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
from processData import load_data

np.random.seed(90)

def relu(z):
    return z * (z > 0)

def drelu(z):
    return (z > 0)

def softmax(z):
    tmp = z - np.max(z, axis=1, keepdims=True)
    return np.exp(tmp) / np.sum(np.exp(tmp), axis=1, keepdims=True)

def cross_entropy(Y_hat, Y):
    delta = 1e-7 # 用于防止overflow
    return -np.sum(Y * np.log(Y_hat + delta)) / len(Y)

def accuracy(Y_hat, Y):
    return np.sum(np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1)) / len(Y)

def load_model(model_name):
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    return model 

def save_model(model):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

class MyModel(object):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        '''
        三层多层感知机分类器: 1 input layer, 2 hidden layer, 1 output layer
        '''
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim 

        self.init_parameters()
        
        self.a = [None for _ in range(4)]
        self.z = [None for _ in range(3)]

        self.init_log()
    
    def init_parameters(self):
        self.W1 = np.random.normal(size=(self.hidden1_dim, self.input_dim)) * np.sqrt(2 / self.input_dim)
        self.b1 = np.zeros((self.hidden1_dim, 1))
        self.dW1 = None 
        self.db1 = None
        
        self.W2 = np.random.normal(size=(self.hidden2_dim, self.hidden1_dim)) * np.sqrt(2 / self.hidden1_dim)
        self.b2 = np.zeros((self.hidden2_dim, 1))
        self.dW2 = None 
        self.db2 = None
        
        self.W3 = np.random.normal(size=(self.output_dim, self.hidden2_dim)) * np.sqrt(2 / self.hidden2_dim)
        self.b3 = np.zeros((self.output_dim, 1))
        self.dW3 = None 
        self.db3 = None 
    
    def init_log(self):
        self.loss_train_list = []
        self.loss_test_list = []
        self.accuracy_test_list = []
    
    def forward(self, X):
        # input
        self.a[0] = X
        # H1
        self.z[0] = np.dot(self.a[0], self.W1.T) + self.b1.T
        self.a[1] = relu(self.z[0])
        # H2
        self.z[1] = np.dot(self.a[1], self.W2.T) + self.b2.T
        self.a[2] = relu(self.z[1])
        # output
        self.z[2] = np.dot(self.a[2], self.W3.T) + self.b3.T
        self.a[3] = softmax(self.z[2])
        return self.a[3]
        
    
    def backprop(self, Y, rate, weight_decay):
        # batch_size, hidden2_dim, hidden1_dim, output_dim: 10 
        dl3 = self.a[3] - Y  # batch_size*10
        self.dW3 = np.dot(dl3.T, self.a[2]) + weight_decay * self.W3 # 10*hidden2_dim 
        self.db3 = np.sum(dl3.T, axis=1, keepdims=True) # 10*1
        
        dl2 = np.dot(dl3, self.W3) * drelu(self.z[1])   # batch_size*hidden2_dim
        self.dW2 = np.dot(dl2.T, self.a[1]) + weight_decay * self.W2 # hidden2_dim*hidden1_dim
        self.db2 = np.sum(dl2.T, axis=1, keepdims=True) # hidden2_dim*1
        
        dl1 = np.dot(dl2, self.W2) * drelu(self.z[0])   # batch_size*hidden1_dim
        self.dW1 = np.dot(dl1.T, self.a[0]) + weight_decay * self.W1 # hidden1_dim*784
        self.db1 = np.sum(dl1.T, axis=1, keepdims=True) # hidden1_dim*1
        
        # 梯度更新
        self.W1 -= rate * self.dW1
        self.W2 -= rate * self.dW2 
        self.W3 -= rate * self.dW3 

        self.b1 -= rate * self.db1
        self.b2 -= rate * self.db2 
        self.b3 -= rate * self.db3 

        # 梯度清零
        self.dW1 = None 
        self.dW2 = None
        self.dW3 = None 

        self.db1 = None 
        self.db2 = None 
        self.db3 = None 
    
    
    def predict(self, X):
        Y_hat = self.forward(X)
        return Y_hat 
    
    def get_batchs(self, total_num, batch_size):
        idx = np.random.permutation(total_num)
        return [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]

    def optimize_train(self, X_train, Y_train, epochs, batch_size, lr=0.2, lr_decay_rate=1, weight_decay=0.1, print_log=False):
        " 只用来训练模型，不进行过程模型metrics记录和可视化"
        loss_train_past = loss_train_new = np.inf
        for epoch in range(epochs):
            # mini-batch 
            batchs = self.get_batchs(len(X_train), batch_size)
            for i, batch_idxs in enumerate(batchs):
                X_batch, Y_batch = X_train[batch_idxs], Y_train[batch_idxs]
                self.forward(X_batch)
                self.backprop(Y_batch, lr / batch_size, weight_decay)
                
                accuracy_train, loss_train = self.evaluate(X_train, Y_train)

            if print_log:
                print(' Epoch: {} | train loss: {:.4f} | accuracy: {:.4f}'.format(epoch+1, loss_train, accuracy_train))
            
            loss_train_past = loss_train_new
            loss_train_new = loss_train
           
            # 学习率随epoch下降
            lr *= 1 / (1 + lr_decay_rate * epoch)

            # 设置提前停止条件
            if abs(loss_train_new- loss_train_past) < 1e-3:
                break 

    def optimize_visualize(self, X_train, Y_train, X_valid, Y_valid, epochs, batch_size, lr=0.2, decay_rate=1, weight_decay=0.1, print_log=False):
        " 用来训练模型，同时进行过程模型metrics记录和可视化"

        for epoch in range(epochs):
            # mini-batch 
            batchs = self.get_batchs(len(X_train), batch_size)
            for i, batch_idxs in enumerate(batchs):
                X_batch, Y_batch = X_train[batch_idxs], Y_train[batch_idxs]
                self.forward(X_batch)
                self.backprop(Y_batch, lr / batch_size, weight_decay)
                
                accuracy_train, loss_train = self.evaluate(X_batch, Y_batch)
                
                if print_log:
                    print('Epoch: {} | batch: {} | train loss: {:.4f} | accuracy: {:.4f}'.format(epoch + 1, i + 1, loss_train, accuracy_train))
            
            accuracy_train, loss_train = self.evaluate(X_train, Y_train)
            accuracy_valid, loss_valid = self.evaluate(X_valid, Y_valid)

            # 记录训练过程中的模型性能
            self.accuracy_test_list.append(accuracy_valid)
            self.loss_train_list.append(loss_train)
            self.loss_test_list.append(loss_valid)

            # 学习率随epoch下降 
            lr = 1 / (1 + decay_rate * epoch) * lr 

            # 提前终止训练：如果相邻两个epoch得到的模型已经收敛到设定精度1e-3
            if len(self.loss_train_list) >= 2 and abs(self.loss_train_list[-1] - self.loss_train_list[-2]) < 1e-3:
                self.plot_loss(epoch + 1)
                self.plot_val_acc(epoch + 1)
                break 
            
            # 每5个epoch可视化一次
            if (epoch+1) % 5 == 0 and print_log:
                self.plot_loss(epoch + 1)
                self.plot_val_acc(epoch + 1)

    def plot_loss(self, epoch):
        '''
        visualize the loss of train and val set during training
        '''
        epochs = list(range(1, len(self.loss_train_list) + 1))
        plt.plot(epochs, self.loss_train_list, '.-', label='Train Loss', color='b')
        plt.plot(epochs, self.loss_test_list, '.-', label='Val Loss', color='r')
        plt.legend(['Train Loss', 'Val Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train&Val Loss during Epochs')
        image_name = 'Epoch{}-loss.jpg'.format(epoch)
        plt.savefig(image_name)
        plt.close()
        #plt.show()

    def plot_val_acc(self, epoch):
        '''
        visualize the accuracy of val set during training
        '''
        epochs = list(range(1, len(self.accuracy_test_list) + 1))
        plt.plot(epochs, self.accuracy_test_list, '.-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Val Accuracy')
        plt.title('Val Accuracy during Epochs')
        image_name = 'Epoch{}-val_acc.jpg'.format(epoch)
        plt.savefig(image_name)
        plt.close()
        #plt.show()

    def evaluate(self, X, Y):
        Y_hat = self.predict(X)
        acc = accuracy(Y_hat, Y)
        loss = cross_entropy(Y_hat, Y)
        return acc, loss 

    def precision(self, X, Y):
        '''
        compute precisions of all 10 classes
        precision = TP / (TP + FP)
        '''
        Y_hat = self.predict(X)
        true_labels = np.argmax(Y, axis=1)
        pred_labels = np.argmax(Y_hat, axis=1)
        precision_dict = dict(zip(list(range(10)), [0 for _ in range(10)]))
        for k in range(10):
            TP = sum((true_labels == k) & (pred_labels == k))
            precision_dict[k] = TP / sum(pred_labels == k)
        return precision_dict
        
if __name__ == "__main__":
    epochs = 15
    batch_size = 300
    decay_rate = 0.5
    lr, hidden1_dim, hidden2_dim, weight_decay=0.1, 256, 256, 0.001

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test, classes_num, input_dim = load_data()

    model = MyModel(input_dim, hidden1_dim, hidden2_dim, classes_num)

    model.optimize_visualize(X_train, Y_train, X_valid, Y_valid, epochs, batch_size, lr, decay_rate, weight_decay, print_log=True)

    # test evaluation 
    precision = model.precision(X_test, Y_test)
    print('Precision of test set: ', precision)

