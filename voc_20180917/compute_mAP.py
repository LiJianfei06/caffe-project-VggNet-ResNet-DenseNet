
from voc_eval import voc_eval



classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]



if __name__ == '__main__':
    AP_sum=0
    for i in range(len(classes)):
        AP=voc_eval('./results/voc_{}.txt', '/home/lijianfei/datasets/VOCdevkit/VOC2007/Annotations/{}.xml', '/home/lijianfei/datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt', classes[i], '.')[-1]
        AP_sum+=AP
        print classes[i],':  ',AP

    mAP=AP_sum/len(classes)
    print "mAP:",mAP

