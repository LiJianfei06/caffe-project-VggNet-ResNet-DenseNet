#include "classification_so.hpp"  
#include <vector>  
using namespace std;  
int main(int argc, char** argv) {
    
    clock_t start_time1,end_time1,start_time2,end_time2;
    
    ::google::InitGoogleLogging(argv[0]);
  
    string model_file   = "/home/ljf/caffe-master/examples/ljftest_alphabet/deploy.prototxt";
    string trained_file = "/home/ljf/caffe-master/examples/ljftest_alphabet/caffe_ljftest_train_iter_50000.caffemodel";
    string mean_file    = "/home/ljf/caffe-master/examples/ljftest_alphabet/imagenet_mean.binaryproto";
    string label_file   = "/home/ljf/caffe-master/examples/ljftest_alphabet/test/labels.txt";
    start_time1 = clock();
    Classifier classifier(model_file, trained_file, mean_file, label_file);
    end_time1 = clock();
    double seconds1 = (double)(end_time1-start_time1)/CLOCKS_PER_SEC;
    std::cout<<"init time="<<seconds1<<"s"<<std::endl;
  
    string file = "/home/ljf/caffe-master/examples/ljftest_alphabet/test/image/875.jpg";
  
    std::cout << "---------- Prediction for "
              << file << " ----------" << std::endl;
  
    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    start_time2 = clock();
    std::vector<Prediction> predictions = classifier.Classify(img);
    end_time2 = clock();
    double seconds2 = (double)(end_time2-start_time2)/CLOCKS_PER_SEC;
    std::cout<<"classify time="<<seconds2<<"s"<<std::endl;
  
    /* Print the top N predictions. */
    for (size_t i = 0; i < predictions.size(); ++i) {
      Prediction p = predictions[i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \""<< p.first << "\"" << std::endl;
    }
  }