

ljftest_alphabet 		小网络，	 精度能达到95.72%,放到caffe-master/examples目录下就可以了(62个字符，0~9，A~Z,a~z)
ljftest_alphabet_VggNet     	用VggNet训练，这儿精度能达到97.64%，放到caffe-master/examples目录下就可以了(62个字符，0~9，A~Z,a~z)
ljftest_alphabet_VggNet_BN     	用VggNet训练，每个卷积层后加上BN精度能达到98.72%，虽然显存耗费也多，放到caffe-master/examples目录下就可以了(62个字符，0~9，A~Z,a~z)
ljftest_alphabet_ResNet     	用ResNet训练，精度能达到99.26%，也就是0和o判错可能性大，放到caffe-master/examples目录下就可以了(62个字符，0~9，A~Z,a~z)
ljftest_cifar10_VggNet     	用VggNet训练，精度能达到88.54%，cifar10
ljftest_cifar10_VggNet_BN     	用VggNet训练，每个卷积层后加上BN精度能达到90.00%，cifar10
ljftest_cifar10_ResNet     	用ResNet 训练，精度能达到91.16%，cifar10
ljftest_cifar10_用WRN     	用WRN 训练，精度能达到92.21%，cifar10
ljftest_cifar10_DenseNet     	用DenseNet 训练，精度能达到91.04%，cifar10

