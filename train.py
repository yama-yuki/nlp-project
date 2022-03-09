import os
import csv
import math
import numpy as np
import random
from model import DNN
​
def load_data(fname):
    with open(fname,'r',encoding='utf-8') as fr:
        datas = csv.reader(fr)
        datas_list = []
        for data in datas:
            datas_list.append(list(map(int,data)))
    return np.array(datas_list)/255
​
def one_hot(data_size,num_hot,num_class):
    one_hot = np.zeros((data_size,num_class))
    for i in range(data_size):
        one_hot[i][num_hot]=1
    return one_hot
​
def batcher(data_x,data_y,minibatch_size,shuffl=True):
    minbatch_x = np.empty((0,len(data_x[0])))
    minbatch_y = np.empty((0,len(data_y[0])))
    index_id = list(range(len(data_x)))
    if shuffl:
        random.shuffle(index_id)
    for i,id in enumerate(index_id):
        minbatch_x = np.append(minbatch_x,data_x[id,:].reshape(1,len(data_x[id,:])),axis=0)
        minbatch_y = np.append(minbatch_y,data_y[id,:].reshape(1,len(data_y[id,:])),axis=0)
        if len(minbatch_x)==minibatch_size or i==len(data_x)-1:
            yield minbatch_x,minbatch_y
            minbatch_x = np.empty((0,len(data_x[0])))
            minbatch_y = np.empty((0,len(data_y[0])))
​
if __name__=='__main__':
    dirpath = os.path.dirname(__file__)
    n_input_unit = 784
    n_hidden_unit = 100
    n_output_unit = 10
    lr = 0.01
    minibatch = 200
    epoch = 100
​
    train_X = np.empty((0,n_input_unit))
    train_Y = np.empty((0,n_output_unit))
    test_X = np.empty((0,n_input_unit))
    test_Y = np.empty((0,n_output_unit))
​
​
    for i in range(10):
        print('load data{0}.csv'.format(i))
        fname_train = os.path.join(dirpath,'train/{0}.csv'.format(i))
        fname_test = os.path.join(dirpath,'test/{0}.csv'.format(i))
        train_x = load_data(fname_train)
        train_y = one_hot(len(train_x),i,n_output_unit)
        test_x = load_data(fname_test)
        test_y = one_hot(len(test_x),i,n_output_unit)
        train_X =  np.append(train_X,train_x,axis=0)
        train_Y = np.append(train_Y,train_y,axis=0)
        test_X =  np.append(test_X,test_x,axis=0)
        test_Y = np.append(test_Y,test_y,axis=0)
​
    '''
    fname1_train = os.path.join(dirpath,'train/1.csv')
    fname7_train = os.path.join(dirpath,'train/7.csv')
    fname1_test = os.path.join(dirpath,'test/1.csv')
    fname7_test = os.path.join(dirpath,'test/7.csv')
    train1_x = load_data(fname1_train)
    train1_y = one_hot(len(train1_x),0,n_output_unit)
    train7_x = load_data(fname7_train)
    train7_y = one_hot(len(train7_x),1,n_output_unit)
    test1_x = load_data(fname1_test)
    test1_y = one_hot(len(test1_x),0,n_output_unit)
    test7_x = load_data(fname7_test)
    test7_y = one_hot(len(test7_x),1,n_output_unit)
    train_X =  np.append(train1_x,train7_x,axis=0)
    train_Y = np.append(train1_y,train7_y,axis=0)
    test_X =  np.append(test1_x,test7_x,axis=0)
    test_Y = np.append(test1_y,test7_y,axis=0)
    '''
​
    m = math.ceil(len(train_X)/minibatch)
    #train
    model = DNN(n_input_unit,n_hidden_unit,n_output_unit,lr)
    for ep in range(epoch):
        loss = 0
        accuracy = 0
        for input_x,label_y in batcher(train_X,train_Y,minibatch):
            u_ih,z_ih,u_oh,pre_y = model.forward(input_x)
            accuracy += model.accuracy(pre_y,label_y)
            loss += model.cross_entropy_loss(pre_y,label_y)
            loss_W_ih_delta,loss_B_ih_delta,loss_W_ho_delta,loss_B_ho_delta = model.backward(input_x,u_ih,z_ih,pre_y,label_y)
            model.update(loss_W_ih_delta,loss_B_ih_delta,loss_W_ho_delta,loss_B_ho_delta)
        print('Train | epoch:{0} | loss:{1} | accuracy:{2}'.format(ep+1,loss/m,accuracy/m))
​
    #test
    _,_,_,pre_y = model.forward(test_X)
    accuracy = model.accuracy(pre_y,test_Y)
    loss = model.cross_entropy_loss(pre_y,test_Y)
    print('Teat | loss:{0} | accuracy:{1}'.format(loss,accuracy))
