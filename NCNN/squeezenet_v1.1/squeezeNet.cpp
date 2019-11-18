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
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                count += 1;
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
        printf("feature_number:%d\n" , count);
    }
}

int main ()
{
    cv::Mat img = cv::imread("plane.jpg");
    int w = img.cols;
    int h = img.rows;
    printf("w:%d, h:%d\n",w,h);

    // subtract 128, norm to -1 ~ 1
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_GRAY, w, h, 227, 227);
    float mean[1] = { 128.f };
    float norm[1] = { 1/128.f };
    in.substract_mean_normalize(mean, norm);

    ncnn::Net net;
    net.load_param("caffemodel/squeezenet.param");
    net.load_model("caffemodel/squeezenet.bin");

    // time test 
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);


    ncnn::Mat feat;
    int count = 0;
    while(count <1000){

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);

    ex.input("data", in);

    //ncnn::Mat feat;
    ex.extract("prob", feat);

    count++;}

    gettimeofday(&t2,NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    printf("Use Time:%f s\n",timeuse);
  
    pretty_print(feat);

    return 0;
}
