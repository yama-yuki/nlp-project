import numpy as np
import os,sys
import random
​
def get_weight(n_input,n_output):
    weight = np.random.randn(n_input,n_output)
    #weight = np.zeros((n_input,n_output))
    return weight
def get_bias(n_output):
    bias = np.random.randn(1,n_output)
    #bias = np.zeros((1,n_output))
    return bias
​
class DNN(object):
​
    def __init__(self,n_input_unit,n_hidden_unit,n_output_unit,lr):
        self.W_ih = get_weight(n_input_unit,n_hidden_unit)
        self.B_ih = get_bias(n_hidden_unit)
        self.W_ho = get_weight(n_hidden_unit,n_output_unit)
        self.B_ho = get_bias(n_output_unit)
        self.learn_rate = lr
​
    def linear(self,x,weight,bias):
        return np.dot(x,weight)+bias
​
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
​
    def sigmoid_delta(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
​
    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
​
    def forward(self,x):
        u_ih = self.linear(x,self.W_ih,self.B_ih)
        z_ih = self.sigmoid(u_ih)
        u_ho = self.linear(z_ih,self.W_ho,self.B_ho)
        y = self.softmax(u_ho)
        return u_ih,z_ih,u_ho,y
​
    def cross_entropy_loss(self,pre_y,label_y):
        e = 0.0000001
        loss = -np.sum(label_y*np.log(pre_y+e))/len(label_y)
        return loss
​
    def backward(self,input_x,u_ih,z_ih,pre_y,label_y):
        delta_o = pre_y-label_y
        loss_W_ho_delta = np.dot(z_ih.T,delta_o)
        loss_B_ho_delta = np.sum(delta_o,axis=0,keepdims=True)/len(label_y)
​
        delta_i = np.dot(delta_o,self.W_ho.T)*self.sigmoid_delta(u_ih)
        loss_W_ih_delta = np.dot(input_x.T,delta_i)
        loss_B_ih_delta = np.sum(delta_i,axis=0,keepdims=True)/len(label_y)
        return loss_W_ih_delta,loss_B_ih_delta,loss_W_ho_delta,loss_B_ho_delta
​
    def update(self,loss_W_ih_delta,loss_B_ih_delta,loss_W_ho_delta,loss_B_ho_delta):
        self.W_ih -= self.learn_rate*loss_W_ih_delta
        self.B_ih -= self.learn_rate*loss_B_ih_delta
        self.W_ho -= self.learn_rate*loss_W_ho_delta
        self.B_ho -= self.learn_rate*loss_B_ho_delta
​
    def accuracy(self,pre_y,label_y):
        correct = np.sum(np.argmax(pre_y,axis=1)==np.argmax(label_y,axis=1))
        accuracy = correct/len(label_y)
        return accuracy
