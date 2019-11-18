#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <net.h>
#include <mat.h>

#include<sys/time.h>

using namespace cv;
using namespace std;
using namespace ncnn;

int main()
{
	//read image
	//wget https://raw.githubusercontent.com/biubug6/Face-Detector-1MB-with-landmark/master/img/sample.jpg
	std::string imgPath = "sample.jpg";
	cv::Mat img = cv::imread(imgPath.c_str());
	if(!img.data)
    	printf("load error");

    ncnn::Net Net;
	Net.load_param("./models/retina.param");
	Net.load_model("./models/retina.bin");
	printf("load model ok.\n");

	//float _mean_val[3] = { 104.f, 117.f, 123.f };
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 300, 300);
    cv::resize(img, img, cv::Size(300, 300));
	float pixel_mean[3] = {0, 0, 0};
	float pixel_std[3] = {1, 1, 1};
	in.substract_mean_normalize(0, 0);

    // time test 
    struct timeval t1,t2;
    double timeuse;
    double timeuse_warmup;
    gettimeofday(&t1,NULL);
    int count = 0;

    while(count <23){
	ncnn::Extractor ex = Net.create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(4);
	ex.input("data", in);
	ncnn::Mat out, out1, out2;

	// class
	ex.extract("face_rpn_cls_prob_reshape_stride8", out);

	// bbox
	ex.extract("face_rpn_bbox_pred_stride8", out1);

	//landmark
	ex.extract("face_rpn_landmark_pred_stride8", out2);

    count++;
    if(count == 3) 
    {
        gettimeofday(&t2,NULL);
        timeuse_warmup = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    }
    
	}

    gettimeofday(&t2,NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    printf("Use Warm Time:%f s\n",timeuse_warmup);
    printf("Use Time:%f s\n",(timeuse-timeuse_warmup) / 20.0);


	return 0;

}