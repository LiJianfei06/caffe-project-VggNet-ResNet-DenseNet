# -*- coding: UTF-8 -*-
import sys
sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import os  
import cPickle  
  
import numpy as np  
import sklearn  
import sklearn.linear_model  
from sklearn.model_selection import train_test_split
import lmdb  
import caffe  
  
def unpickle(file):  
    fo = open(file, 'rb')  
    dict = cPickle.load(fo)  
    fo.close()  
    return dict  
  
#调用sklearn对数据进行shuffle操作  
def shuffle_data(data, labels):  
    data, _, labels, _ = train_test_split(  
        data, labels, test_size=0.0, random_state=42  
    )  
    return data, labels  
  
def load_data(train_file):  
    d = unpickle(train_file)  
    data = d['data']  
    fine_labels = d['fine_labels']  
    length = len(d['fine_labels'])  
  
    data, labels = shuffle_data(  
        data,  
        np.array(fine_labels)  
    )  
    return (  
        data.reshape(length, 3, 32, 32),  
        labels  
    )  
  
if __name__=='__main__':  
    cifar_python_directory = os.path.abspath('cifar-100-python')  
  
    print('Converting...')  
    cifar_caffe_directory=os.path.abspath('train_lmdb')  
    if not os.path.exists(cifar_caffe_directory):  
  
        X,y_f=load_data(os.path.join(cifar_python_directory, 'train'))  
        Xt,yt_f=load_data(os.path.join(cifar_python_directory, 'test'))  
  
        print('Data is fully loaded,now truly convertung.')  
          
        #lmdb操作,将数据写入数据库  
        env=lmdb.open(cifar_caffe_directory,map_size=50000*1000*5)  
        txn=env.begin(write=True)  
        count=0  
        for i in range(X.shape[0]):  
            datum=caffe.io.array_to_datum(X[i],y_f[i])  
            str_id='{:08}'.format(count)  
            txn.put(str_id,datum.SerializeToString())  
  
            count+=1  
            if count%1000==0:  
                print('already handled with {} pictures'.format(count))  
                txn.commit()  
                txn=env.begin(write=True)  
  
        txn.commit()  
        env.close()  
        cifar_test_caffe_directory=os.path.abspath('test_lmdb') 
        env=lmdb.open(cifar_test_caffe_directory,map_size=10000*1000*5)  
        txn=env.begin(write=True)  
        count=0  
        for i in range(Xt.shape[0]):  
            datum=caffe.io.array_to_datum(Xt[i],yt_f[i])  
            str_id='{:08}'.format(count)  
            txn.put(str_id,datum.SerializeToString())  
  
            count+=1  
            if count%1000==0:  
                print('already handled with {} pictures'.format(count))  
                txn.commit()  
                txn=env.begin(write=True)  
  
                txn.commit()  
                env.close()  
            else:  
                print('Conversion was already done. Did not convert twice.')  
