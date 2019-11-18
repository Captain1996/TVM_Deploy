#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <net.h>
#include <mat.h>

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
    cv::Mat img = cv::imread("cat.png");
    int w = img.cols;
    int h = img.rows;
    printf("w:%d, h:%d\n",w,h);

    // subtract 128, norm to -1 ~ 1
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, w, h, 224, 224);
    float mean[3] = { 103.94f, 116.78f, 123.68f };
    float norm[3] = {0.017f,0.017f,0.017f };
    in.substract_mean_normalize(mean, norm);

    ncnn::Net net;
    net.load_param("./MobileNet-Caffe/mobilenet_v2.param");
    net.load_model("./MobileNet-Caffe/mobilenet_v2.bin");

    // time test 
    struct timeval t1,t2;
    double timeuse;
    double timeuse_warmup;
    gettimeofday(&t1,NULL);


    ncnn::Mat feat;
    int count = 0;
    while(count <23){

    ncnn::Extractor ex = net.create_extractor();
    //ex.set_num_threads(4);

    ex.input("data", in);

    //ncnn::Mat feat;
    ex.extract("prob", feat);

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
    printf("Use Time:%f s\n",timeuse);
  
    pretty_print(feat);

    return 0;
}

