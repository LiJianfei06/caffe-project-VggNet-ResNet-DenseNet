#coding:utf-8
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import sys
import cv2
#caffe_root = '../../../'  
sys.path.append("/home/lijianfei/caffe-yolov2/python")
sys.path.append("/home/lijianfei/caffe-yolov2/python/caffe")
import caffe
import math
import os
import shutil
import time


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes_list=[]
for i in range(len(classes)):
    file_name="./results/voc_%s.txt"%str(classes[i])
    if os.path.exists(file_name)==True:
        os.remove(file_name)
    classes_list.append(file_name)


caffe.set_device(0)
caffe.set_mode_gpu()

model_def = './deploy_yolov2.prototxt'
#model_weights = '../convert/yolo.caffemodel'
model_weights = './models_save/yolov2_VOC0712_iter_45000.caffemodel'



net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

dim_in =416
mu = np.array([105, 117, 123])

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255.0)      # rescale from [0, 1] to [0, 255]
#transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_input_scale('data', 0.00390625)

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          dim_in, dim_in)  # image size is 227x227




def det(image, image_id, pic,w_new,h_new,ratio,save_path):
        fp = list(range(0,len(classes_list)))
        for i in range(len(classes_list)):
            fp[i]=open(classes_list[i],"a+") 

	transformed_image = transformer.preprocess('data', image)
	#plt.imshow(image)

	net.blobs['data'].data[...] = transformed_image
        ti1 = time.time()

	### perform classification
	output = net.forward()
        """
        print "conv1:",net.blobs['conv1'].data[...][0][0][:][:]
        print "conv2:",net.blobs['conv2'].data[...][0][0][:][:]
        print "conv3:",net.blobs['conv3'].data[...][0][0][:][:]
        print "conv4:",net.blobs['conv4'].data[...][0][0][:][:]
        print "conv9:",net.blobs['conv9'].data[...][0][0][:][:]
        print "conv14:",net.blobs['conv14'].data[...][0][0][:][:]


        print "conv21:",net.blobs['conv21'].data[...][0][0][:][:]
        print "conv22:",net.blobs['conv22'].data[...][0][0][:][:]
        """

        ti2 = time.time()
        print "forward time cost:",ti2-ti1
	
	res = output['detection_out'][0]  # the output probability vector for the first image in the batch
	
	#print res.shape
	index = 0
	box = []
	boxes = []
	for c in range(res.shape[0]):
		for h in range(res.shape[1]):			
			for w in range(res.shape[2]):
				box.append(res[c][h][w])
			boxes.append(box)
			box = []
	#print boxes

	im = cv2.imread(pic)
	w = im.shape[1]
	h = im.shape[0]
	for box in boxes:
                print "%10s"%classes[int(box[1])],box
		left = (box[3]-box[5]/2.0) * w;
		right = (box[3]+box[5]/2.0) * w;
		#top = (box[4]-box[6]/2.0) *dim_in/h_new* h;
		#bot = (box[4]+box[6]/2.0) *dim_in/h_new* h;
		top = ((box[4]-box[6]/2.0) *dim_in-(dim_in-int(h_new))/2)/h_new* h;
		bot = ((box[4]+box[6]/2.0) *dim_in-(dim_in-int(h_new))/2)/h_new* h;
		if left < 0:
			left = 0
		if right > w:
			right = w
		if top < 0:
			top = 0
		if bot > h:
			bot = h
		color = (255, 242, 35)
		cv2.rectangle(im,(int(left), int(top)),(int(right),int(bot)),color,3)
                cv2.putText(im,str(classes[int(box[1])])+" %.02f"%float(box[2]),(int(left),int(top)),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),1)
                #if int(box[1])==1:
                fp[int(box[1])].write(str(image_id)+' ')  	
                fp[int(box[1])].write("%6f"%float(box[2])+' ')  	
                fp[int(box[1])].write("%6f"%float(left)+' ')  	
                fp[int(box[1])].write("%6f"%float(top)+' ')  	
                fp[int(box[1])].write("%6f"%float(right)+' ')  	
                fp[int(box[1])].write("%6f"%float(bot)+' ')  	
                fp[int(box[1])].write('\n')  	
	#cv2.imshow('src', im)
	#cv2.imwrite('result_pic.jpg', im)
	#cv2.imwrite(save_path+str(image_id)+'.jpg', im)
	#cv2.waitKey()
        for i in range(len(classes_list)):
            fp[i].close() 
	#print 'det'
					

def deal_a_image(image):
    #print np.shape(image)

    max_wh=max(np.shape(image)[0],np.shape(image)[1])
    ratio=1.0*max_wh/dim_in

    #print "ratio:",ratio
    w_new=np.shape(image)[1]/ratio
    h_new=np.shape(image)[0]/ratio

    #image1=cv2.resize(image,(int(w_new),int(h_new)),interpolation=cv2.INTER_NEAREST)
    image1=cv2.resize(image,(int(w_new),int(h_new)),interpolation=cv2.INTER_LINEAR)
    img=np.zeros((dim_in,dim_in,3),np.float)
    img[:][:][:]=0.5

    #img[0:int(h_new),0:int(w_new)]=image1
    img[(dim_in-int(h_new))/2:int(h_new)+(dim_in-int(h_new))/2,(dim_in-int(w_new))/2:int(w_new)+(dim_in-int(w_new))/2]=image1

    return img,w_new,h_new,ratio




def test_a_image(pic):
    image = caffe.io.load_image(pic)
   
    img,w_new,h_new,ratio =deal_a_image(image)
    cv2.imwrite('temp.jpg', img*255)

    det(img, '10001', pic,w_new,h_new,ratio,'./')
    print 'over'


def test_dataset(file_list):

    save_img_place='./save_img/'
    if os.path.exists(save_img_place)==False:
        os.makedirs(save_img_place)
    os.system("rm -rf "+save_img_place+"*")

    cnt=0
    fr = open(file_list)
    read_all=fr.readlines()
    len_all=len(read_all)
    for line in read_all:
        lineArr = line.strip().split('\t')
        cnt+=1
        print str(cnt)+"/"+str(len_all)
        print lineArr[0]
        image = caffe.io.load_image(lineArr[0])
        img,w_new,h_new,ratio =deal_a_image(image)
        det(img, lineArr[0].split('/')[-1].split('.')[0], lineArr[0],w_new,h_new,ratio,save_img_place)


    print 'over'
    fr.close()



if __name__ == '__main__':
    pic = "/home/lijianfei/darknet/data/dog.jpg"
    #pic = "./predictions1.jpg"
    #test_a_image(pic)
    

    file_list="/home/lijianfei/darknet/test_voc.txt"
    test_dataset(file_list)






