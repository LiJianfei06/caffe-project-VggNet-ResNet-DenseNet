# -*- coding: UTF-8 -*-
import sys
sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import lmdb  
import os  
import cv2  
import cPickle  
import caffe  
from caffe.proto import caffe_pb2  
  
def unpickle(file):  
    fo = open(file, 'rb')  
    dict = cPickle.load(fo)  
    fo.close()  
    return dict  
  
if __name__=='__main__':  
    #得到label具体对应的种类  
    meta=unpickle(os.path.join('cifar-10-batches-py', 'batches.meta'))  
    fine_label_names=meta['label_names']  
  
    env=lmdb.open('train_lmdb')  
    txn=env.begin()  
    cursor=txn.cursor()  
    datum=caffe_pb2.Datum()  
  
    i=0  
    for key,value in cursor:  
        datum.ParseFromString(value)  
        if i<50000:  
            data=caffe.io.datum_to_array(datum)  
            label=datum.label  
            img=data.transpose(1,2,0)  
            #图片名字为其类别  
            cv2.imwrite('train/{}.png'.format(fine_label_names[label]+str(i)),img)  
        i+=1  
        print i
  
    env.close()  
print('there are totally {} pictures'.format(i))  
