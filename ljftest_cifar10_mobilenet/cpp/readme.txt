//g++ -o run classification.cpp -I/home/ljf/caffe-master/include/ -I/home/ljf/caffe-master/.build_release/src/ -I/usr/local/include -I/usr/local/cuda/include -I/usr/include -L/home/ljf/caffe-master/build/lib/ -lcaffe -lboost_system -lglog `pkg-config --cflags --libs opencv`     //GPU模式

//g++ -o run classification.cpp -D CPU_ONLY -I/home/ljf/caffe-master/include/ -I/home/ljf/caffe-master/.build_release/src/ -I/usr/local/include -I/usr/local/cuda/include -I/usr/include -L/home/ljf/caffe-master/build/lib/ -lcaffe -lboost_system -lglog `pkg-config --cflags --libs opencv`     // 未知服务器是否有N卡，先cpu跑吧  直接一个.cpp生产可执行文件run



//g++ -std=c++11 -O2 -DDLIB_JPEG_SUPPORT classification_so.cpp -fPIC -shared -I/home/ljf/caffe-master/include/ -I/home/ljf/caffe-master/.build_release/src/ -I/usr/local/include -I/usr/local/cuda/include -I/usr/include -L/home/ljf/caffe-master/build/lib/ -lcaffe -lboost_system -lglog -o libclassification_so.so `pkg-config --cflags --libs opencv`   //生成libclassification_so.so  这里还是供ｃ++调用的


//g++ -std=c++11 -O2 -DDLIB_JPEG_SUPPORT -o main main.cpp -I/home/ljf/caffe-master/include/ -I/home/ljf/caffe-master/.build_release/src/ -I/usr/local/include -I/usr/local/cuda/include -I/usr/include -L. -lclassification_so -L/home/ljf/caffe-master/build/lib/ -lcaffe -lboost_system -lglog `pkg-config --cflags --libs opencv`      //链接生成可执行文件main

//export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH　　　若执行时提示找不到路径就执行这条指令
