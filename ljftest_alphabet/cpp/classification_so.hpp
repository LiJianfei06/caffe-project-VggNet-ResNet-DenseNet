#pragma once
#include "caffe/caffe.hpp"  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <algorithm>  
#include <iosfwd>  
#include <memory>  
#include <string>  
#include <utility>  
#include <vector>  
#include <sstream>  
  
using namespace caffe;  
using namespace std;  
using namespace cv;  
  

typedef std::pair<string, float> Prediction;

class Classifier {
public:
 Classifier(const string& model_file,
            const string& trained_file,
            const string& mean_file,
            const string& label_file);

 std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
 void SetMean(const string& mean_file);

 std::vector<float> Predict(const cv::Mat& img);

 void WrapInputLayer(std::vector<cv::Mat>* input_channels);

 void Preprocess(const cv::Mat& img,
                 std::vector<cv::Mat>* input_channels);

private:
    std::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<string> labels_;
};
