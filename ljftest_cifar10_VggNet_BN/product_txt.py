#coding:utf-8
from numpy import *
import operator
import os
from os import listdir
import os.path
import matplotlib
import matplotlib.pyplot as plt
import re

from PIL import Image
from PIL import ImageFilter,ImageEnhance
  

def generate_txt(str_place,filename_txt):
    fp=open(filename_txt,"w")
    i=0
    for dirpath, dirnames, filenames in os.walk(str_place):
        print "Directory:%s"%dirpath
        for filename in filenames:
            #print i,filename,filename[0:4]
            
            n=0
            if(ord(filename[n])<=57 and ord(filename[n])>=48):
                idx = ord(filename[n])-48
            elif(ord(filename[n])<=90 and ord(filename[n])>=65):            # A....
                idx = ord(filename[n])-65+10
            elif(ord(filename[n])<=122 and ord(filename[n])>=97):           # a....
                idx = ord(filename[n])-97+36            
            

            fp.write(filename+' '+str(idx)+'\n')
            
#            i=i+1
    fp.close()


if __name__ == '__main__':
    generate_txt("/home/ljf/caffe-master/examples/ljftest_alphabet/train/",
                 "/home/ljf/caffe-master/examples/ljftest_alphabet/train.txt")
    
    generate_txt("/home/ljf/caffe-master/examples/ljftest_alphabet/val/",
                 "/home/ljf/caffe-master/examples/ljftest_alphabet/val.txt")

