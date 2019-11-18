#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "./build/install/include/ncnn/net.h"
#include "./build/install/include/ncnn/mat.h"

#include<sys/time.h>

using namespace std;
using namespace cv;
using namespace ncnn;

void pretty_print(const ncnn::Mat& m)
{
    int count = 0;
    float maxnum = 0;
    int index = 0;
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                count += 1;
                printf("%f ", ptr[x]);
                if(ptr[x] >= maxnum) { maxnum = ptr[x]; index = x;}
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
        printf("feature_number:%d\n" , count);
    }
    printf("max num: %f , %d\n", maxnum, index);
}

int main ()
{
	//wget https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/imgs/2.jpg
    cv::Mat img = cv::imread("2.jpg");
    int w = img.cols;
    int h = img.rows;
    printf("w:%d, h:%d\n",w,h);

    // subtract 128, norm to -1 ~ 1
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, w, h);
    float mean[3] = { 127.5f, 127.5f, 127.5f  };
    float norm[3] = {1/127.5f, 1/127.5f, 1/127.5f };
    in.substract_mean_normalize(mean, norm);

    ncnn::Net net;
    net.load_param("./models/symbol_10_320_20L_5scales_v2_deploy.param");
    net.load_model("./models/train_10_320_20L_5scales_v2_iter_1000000.bin");

    // time test 
    struct timeval t1,t2;
    double timeuse;
    double timeuse_warmup;
    gettimeofday(&t1,NULL);


    ncnn::Mat feat;
    int count = 0;
    while(count <23){

    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(4);

    ex.input("data", in);

    //ncnn::Mat feat;
    ex.extract("conv20_3_bbox", feat);

    count++;
    if(count == 3) 
         {
            gettimeofday(&t2,NULL);
            timeuse_warmup = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
         }
    }

    gettimeofday(&t2,NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    printf("Use Time:%f s\n",timeuse_warmup);
    printf("Use Time:%f s\n",(timeuse - timeuse_warmup)/ 20.0);
  
    pretty_print(feat);

    return 0;
}