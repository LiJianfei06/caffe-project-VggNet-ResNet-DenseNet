

ljftest_alphabet 				放到caffe-master/examples目录下就可以了(62个字符，0~9，A~Z,a~z)
ljftest_alphabet_VggNet     	用VggNet训练，这儿精度能提高1.5%左右，放到caffe-master/examples目录下就可以了(62个字符，0~9，A~Z,a~z)
ljftest_alphabet_VggNet_BN     	用VggNet训练，每个卷积层后加上BN精度还能提升，虽然显存耗费也多，放到caffe-master/examples目录下就可以了(62个字符，0~9，A~Z,a~z)
ljftest_alphabet_ResNet     	用ResNet训练，精度能达到99.4%以上，也就是0和o判错可能性大，放到caffe-master/examples目录下就可以了(62个字符，0~9，A~Z,a~z)