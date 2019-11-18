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
	std::string imgPath = "sample.jpg";
	cv::Mat img = cv::imread(imgPath.c_str());

	// scale
    const int max_side = 320;
	float long_side = std::max(img.cols, img.rows);
	float scale = max_side / long_side;
	cv::Mat img_scale;
	cv::Size size = cv::Size(img.cols*scale, img.rows*scale);
	cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));
	printf("load image ok.\n");

    ncnn::Net *Net = new ncnn::Net();
	Net->load_param("./model/face.param");
	Net->load_model("./model/face.bin");
	printf("load model ok.\n");

	float _mean_val[3] = { 104.f, 117.f, 123.f };
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_scale.data, ncnn::Mat::PIXEL_BGR, img_scale.cols, img_scale.rows, img_scale.cols, img_scale.rows);
	in.substract_mean_normalize(_mean_val, 0);

    // time test 
    struct timeval t1,t2;
    double timeuse;
    double timeuse_warmup;
    gettimeofday(&t1,NULL);
    int count = 0;

    while(count <23){
	ncnn::Extractor ex = Net->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(4);
	ex.input("input0", in);
	ncnn::Mat out, out1, out2;

	// loc
	ex.extract("output0", out);

	// class
	ex.extract("530", out1);

	//landmark
	ex.extract("529", out2);

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