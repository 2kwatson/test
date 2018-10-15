
import csv

import numpy as np

from numpy.linalg import inv

#import sys

import math

#import random

 

 

#############################################

#parse data

#initial

data = []

for i in range(18):

        data.append([]) #生成data[[]...[]]  18个[]

n_row = 0

#print("data:",data)

#print("type(data):",type(data))

 

text = open('train.csv','r',encoding='big5')

row = csv.reader(text,delimiter=",")  #在列表中把text的数据按逗号分隔开来

 

print("type(text):",type(text))

print("type(row):",type(row))

 

 

for r in row:

        #0 cols

        if n_row != 0:

                for i in range(3,27): #第三列到第26列

                        if r[i] != "NR":

                                data[(n_row-1)%18].append(float(r[i]))
                                #如果不是NR，就转换成float类型加到data[0]中
                                #相当于一个data[0]中有24个数据，即24个小时的

                        else:

                                data[(n_row-1)%18].append(float(0))
                                #遇到NR就换成0

        n_row = n_row + 1 #最后一个小时的数据加入之后，由data[0]变成data[1]
                          #到n_row成19时，又成为data[0]，即[0]里面存的都是一类
        
#这么说n_row就是总行数，从0加到4320（就是12*20*18）,由于第0行没有数据，故从1开始
        
text.close() #所有数据存完之后，关闭文件

print("data[0]:",data[0][:12])#data[0][:12]指data[0]第一行的前12个元素

print("data[1]:",data[1][:12])#实际一共从data[0]到data[17]，按类把数据存储

 

 

#divide data

x = []

y = []

const_month = 12

src_data_in_month = int(len(data[0]) / const_month)     #480=12*20
#一共240天，每一天每种污染测24下，len(data[x])均为5760，也就是每种污染一年测5760下
#每个月测5760/12=480下
data_per_month = src_data_in_month - 9          #471
#data_per_month是每月实打实训练的数据

#per month

for i in range(12):#i代表月份

        for j in range(data_per_month):#j代表每月实际训练的数据数目471

                x.append([])

                for t in range(18):#t代表污染的种类

                        for s in range(9):

                                x[471*i+j].append(data[t][480*i+j+s])

                y.append(data[9][480*i+j+9])    #s=9

print("after append,x[0]:",x[0][:10])

print("len(x):",len(x))

 

x=np.array(x)

y=np.array(y)

print("len(x):",len(x))

 

x=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)#按列增加的方向拼接
#x.shape是x的行数与列数，shape[0]是行数
#5652行，1列的全1矩阵

print("len(x):",len(x))

w2 = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)

#matmul是矩阵乘法 ，transpose是转置,inv是求逆

################################################

#train 

w = np.zeros(len(x[0])) #163

print("len(x[0]):",len(x[0]))

l_rate=10

repeat=10000

#print("w[0]:",w[0][:10])

#print("w[1]:",w[1][:10])

 

x_t = x.transpose()

s_gra = np.zeros(len(x[0]))

 

for i in range(repeat):

        hypo = np.dot(x,w)#x与w乘法

        loss = hypo - y

        cost = np.sum(loss**2)/len(x)

        cost_a = math.sqrt(cost)

        gra = np.dot(x_t,loss)

        s_gra += gra**2

        ada = np.sqrt(s_gra)

        w = w - l_rate * gra/ada

        print('iteration:%d | Cost: %f ' %(i,cost_a))

 

np.save('model.npy',w)

 

w2 = np.load('model.npy')

 

 

#######################################################

#test

test_x = []

n_row = 0

text = open('test.csv',"r")

row = csv.reader(text, delimiter=",")

 

for r in row:

        if n_row % 18 == 0:

                test_x.append([])

                for i in range(2,11):

                        test_x[n_row//18].append(float(r[i]))

        else:

                for i in range(2,11):

                        if r[i] != "NR":

                                test_x[n_row//18].append(float(r[i]))

                        else:

                                test_x[n_row//18].append(0)

        n_row = n_row + 1

text.close()

test_x = np.array(test_x)

test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x),axis=1)

 

 

#######################################################3

#test

ans = []

for i in range(len(test_x)):

        ans.append(["id_"+str(i)])

        a = np.dot(w,test_x[i])

        ans[i].append(a)

#############

'''

        a = np.dot(w,test_x[i])

        a2 = np.dot(w2,test_x[i])

        loss = a - a2

        cost = np.sum(loss**2)/len(test_x[i])

        cost_a = math.sqrt(cost)

        print("test Cost:%f"%(cost_a))

'''

########################

filename = "predict.csv"

text = open(filename,"w+")

s = csv.writer(text,delimiter=',',lineterminator='\n')

s.writerow(["id","value"])

for i in range(len(ans)):

        s.writerow(ans[i])

text.close()

 

 

 
