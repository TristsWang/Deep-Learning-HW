from encodings import normalize_encoding
from torchvision import datasets
import numpy as np 
import struct


def download_data():
    train = datasets.FashionMNIST('./data', train=True, download=True)
    test = datasets.FashionMNIST('./data', train=False, download=True)

def read_datafile(image_path, label_path):
    with open(label_path, 'rb') as f:
        _, _ = struct.unpack('>II', f.read(8))
        label_data = np.fromfile(f, dtype=np.int8)
    with open(image_path, 'rb') as f:
        _, _, rows, cols = struct.unpack('>IIII', f.read(16))
        image_data = np.fromfile(f, dtype=np.uint8).reshape(len(label_data), rows, cols)
    return image_data, label_data 

def load_data():
    import os
    print(os.getcwd())
    path = './data/FashionMNIST/raw/'
    X_train, Y_train = read_datafile(path + 'train-images-idx3-ubyte', path + 'train-labels-idx1-ubyte')
    X_test, Y_test = read_datafile(path + 't10k-images-idx3-ubyte', path + 't10k-labels-idx1-ubyte')

    input_dim = X_train.shape[-1] * X_train.shape[-2]
    # normalize 
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    
    # flatten
    X_train = X_train.reshape(len(X_train), input_dim)
    X_test = X_test.reshape(len(X_test), input_dim)
    
    # to one-hot 
    Y_train = one_hot(Y_train)
    Y_test = one_hot(Y_test)

    classes_num = 10
    
    # 划分训练集和测试集
    valid_portion = 0.1
    valid_len = int(valid_portion * X_train.shape[0])
    
    index = [i for i in range(X_train.shape[0])]
    np.random.shuffle(index)
    
    X_valid, Y_valid = X_train[index[0:valid_len]], Y_train[index[0:valid_len]]
    X_train, Y_train = X_train[index[valid_len:]], Y_train[index[valid_len:]]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, classes_num, input_dim

def one_hot(labels):
    return np.eye(10)[labels]

def normalize(X):
    return 2 * (X / 255 - 0.5)
