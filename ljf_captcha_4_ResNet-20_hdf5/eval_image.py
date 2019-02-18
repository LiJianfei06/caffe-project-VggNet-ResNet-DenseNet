# -*- coding: UTF-8 -*-
"""
Created on Wed Aug 15 22:31:23 2018

python eval_image.py --proto deploy_MnasNet.prototxt --model ./model_save/MnasNet_model_cat_dog_iter_10000.caffemodel  --image ./cat.jpg

@author: lijianfei
"""
from __future__ import print_function
import argparse
import numpy as np
import sys
import os
import cv2
sys.path.append("/home/lijianfei/caffe-master-ljf/python")
sys.path.append("/home/lijianfei/caffe-master-ljf/python/caffe")
import caffe
import time
from PIL import Image

caffe.set_device(0)
caffe.set_mode_cpu()

    


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate pretrained mobilenet models')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--image', dest='image',
                        help='path to color image', type=str)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()

CHAR_SET_LEN=4

# 向量转回文本  
def vec2text(vec):  
    text=[]  
    for c in vec:  
        #print (c)
        #char_at_pos = i #c/63  
        char_idx = c #% CHAR_SET_LEN  
        if char_idx < 10:  
            char_code = char_idx + ord('0')  
        elif char_idx <36:  
            char_code = char_idx - 10 + ord('A')  
        elif char_idx < 62:  
            char_code = char_idx-  36 + ord('a')  
        else:  
            raise ValueError('error')  
        text.append(chr(char_code))  
    return "".join(text)  


def eval():
    nh, nw = 32, 72
    #img_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)


    #net = caffe.Net(args.proto, args.model, caffe.TEST)
    net = caffe.Net("deploy_ResNet_20.prototxt", "./model_save/cifar10_ResNet_20_iter_64000.caffemodel", caffe.TEST)
    #sys.exit()
    #im = caffe.io.load_image(args.image)
    for dirpath, dirnames, filenames in os.walk("/home/lijianfei/datasets/captcha_example/test/"):
        for ii, filename in enumerate(filenames):
            im = caffe.io.load_image("/home/lijianfei/datasets/captcha_example/test/"+filename,False)
            #h, w, _ = im.shape
            #if h < w:
            #    off = (w - h) / 2
            #    im = im[:, off:off + h]
            #else:
            #    off = (h - w) / 2
            #    im = im[off:off + h, :]
            #im = caffe.io.resize_image(im, [nh, nw])

            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))  # row to col
            #transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
            transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
            #transformer.set_mean('data', img_mean)
            transformer.set_input_scale('data', 0.00390625)


            net.blobs['data'].reshape(1, 1, nh, nw)
            net.blobs['data'].data[...] = transformer.preprocess('data', im)
            #net.blobs['data'].data[...]=image_cv
            for i in range(1):
                start = time.clock()
                out = net.forward()
                elapsed = (time.clock() - start)
                print("Time used:",elapsed," s")

            vec_list=[]
            prob1 = out['prob1']
            prob1 = np.squeeze(prob1)
            idx1 = np.argsort(-prob1)
            vec_list.append(idx1[0])

            prob2 = out['prob2']
            prob2 = np.squeeze(prob2)
            idx2 = np.argsort(-prob2)
            vec_list.append(idx2[0])

            prob = out['prob3']
            prob = np.squeeze(prob)
            idx = np.argsort(-prob)
            vec_list.append(idx[0])


            prob = out['prob4']
            prob = np.squeeze(prob)
            idx = np.argsort(-prob)
            vec_list.append(idx[0])


            print ("filename:",filename,", forward:",vec2text(vec_list))


if __name__ == '__main__':
    eval()
