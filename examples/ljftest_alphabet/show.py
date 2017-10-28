# -*- coding: UTF-8 -*-
#!/usr/bin/env python  
#coding=utf-8  
import numpy as np  
import pylab  
import matplotlib.pyplot as plt  
from PIL import Image  
caffe_root = '/home/ljf/caffe-master/'  
import sys  
sys.path.insert(0, caffe_root + 'python')  
import caffe  
  
#caffe.set_mode_cpu()  
  
caffe.set_device(0)  
caffe.set_mode_gpu()  
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.  
import os  
if os.path.isfile(caffe_root + '/home/ljf/caffe-master/examples/ljftest_alphabet/caffe_ljftest_train_iter_100000.caffemodel'):  
    print 'CaffeNet found.'  
else:  
    print 'Downloading pre-trained CaffeNet model...'  
#../scripts/download_model_binary.py ../models/bvlc_reference_caffenet  
  
model_def = caffe_root + 'examples/ljftest_alphabet/deploy.prototxt'  
model_weights = caffe_root + 'examples/ljftest_alphabet/caffe_ljftest_train_iter_50000.caffemodel'  
  
net = caffe.Net(model_def,      # defines the structure of the model  
                model_weights,  # contains the trained weights  
                caffe.TEST)     # use test mode (e.g., don't perform dropout)  
# load the mean ImageNet image (as distributed with Caffe) for subtraction  
mu = np.load(caffe_root + 'examples/ljf_CaffeNet_test/imagenet_mean.npy')  
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values  
print 'mean-subtracted values:', zip('BGR', mu)  
#sys.exit(0)
# create transformer for the input called 'data'  
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  
#print net.blobs['data'].data.shape
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension  
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel  
transformer.set_raw_scale('data', 1)      # rescale from [0, 1] to [0, 255]  
#transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR  
# set the size of the input (we can skip this if we're happy  
#  with the default; we can also change it later, e.g., for different batch sizes)  
net.blobs['data'].reshape(50,        # batch size  
                          3,         # 3-channel (BGR) images  
                          32, 32)  # image size is 227x227  
image = caffe.io.load_image(caffe_root + 'examples/ljftest_alphabet/test/image/875.jpg')  
transformed_image = transformer.preprocess('data', image)  
plt.imshow(image)  
# copy the image data into the memory allocated for the net  
net.blobs['data'].data[...] = transformed_image  

### perform classification  
output = net.forward()  
  
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch  
  
print 'predicted class is:', output_prob.argmax()  
  
  
# load ImageNet labels  
labels_file = caffe_root + 'examples/ljftest_alphabet/test/labels.txt'  
#if not os.path.exists(labels_file):  
#    !../data/ilsvrc12/get_ilsvrc_aux.sh  
      
labels = np.loadtxt(labels_file, str, delimiter='\t')  
  
print 'output label:', labels[output_prob.argmax()]  
  
# sort top five predictions from softmax output  
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items  
  
print 'probabilities and labels:'  
zip(output_prob[top_inds], labels[top_inds])  
# for each layer, show the output shape  

for layer_name, blob in net.blobs.iteritems():  
    print layer_name + '\t' + str(blob.data.shape)  
for layer_name, param in net.params.iteritems():  
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)  
def vis_square(data):  
    """Take an array of shape (n, height, width) or (n, height, width, 3) 
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""  
      
    # normalize data for display  
    data = (data - data.min()) / (data.max() - data.min())  
      
    # force the number of filters to be square  
    n = int(np.ceil(np.sqrt(data.shape[0])))  
    padding = (((0, n ** 2 - data.shape[0]),  
               (0, 1), (0, 1))                 # add some space between filters  
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)  
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)  
      
    # tile the filters into an image  
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))  
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])  
    plt.imshow(data); plt.axis('off')  
    pylab.show()  
# the parameters are a list of [weights, biases]  
print 'the first layer filters,conv1'
filters = net.params['conv1'][0].data  
vis_square(filters.transpose(0, 2, 3, 1))  

print "data"
feat = net.blobs['data'].data[0, :36]  
vis_square(feat) 

print "conv1"
feat = net.blobs['conv1'].data[0, :36]  
vis_square(feat)  
  
print "conv2"
feat = net.blobs['conv2'].data[0, :36]  
vis_square(feat) 

print "conv3"
feat = net.blobs['conv3'].data[0, :36]  
vis_square(feat) 
#
#print "conv4"
#feat = net.blobs['conv4'].data[0, :36]  
#vis_square(feat) 
#
#print "conv5"
#feat = net.blobs['conv5'].data[0, :36]  
#vis_square(feat) 
#
#print "conv6"
#feat = net.blobs['conv6'].data[0, :36]  
#vis_square(feat) 

print "fc65"
feat = net.blobs['fc65'].data[0]  
plt.subplot(2, 1, 1)  
plt.plot(feat.flat)  
plt.subplot(2, 1, 2)  
_ = plt.hist(feat.flat[feat.flat > 0], bins=100) 

print "prob" 
feat = net.blobs['prob'].data[0]  
plt.figure(figsize=(15, 3))  
plt.plot(feat.flat)  
pylab.show()  