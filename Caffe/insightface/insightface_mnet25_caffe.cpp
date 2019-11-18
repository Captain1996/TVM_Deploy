#define CPU_ONLY
#include <fstream>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include "caffe/caffe.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

int main()
{
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	
	caffe::Net<float> *caffe_net_test = nullptr;
	
	caffe_net_test = new caffe::Net<float>("./mnet/mnet.prototxt", caffe::TEST);

	caffe::NetParameter trained_net_param_test;

	if (!caffe::ReadProtoFromBinaryFile("./mnet/mnet.caffemodel", &trained_net_param_test)) {
  		std::cout<<"read trained file failed"<<std::endl;  
		return -1;
  	}
	
	caffe_net_test->CopyTrainedLayersFrom(trained_net_param_test);

	caffe::Blob<float>* input = caffe_net_test->blobs().at(0).get();

	int rwidth = input->width();
 	int rheight = input->height();
 	int rchannels = input->channels();
	
	std::cout<<"rwidth = "<<rwidth<<std::endl;
	std::cout<<"rheight = "<<rheight<<std::endl;
	std::cout<<"rchannels = "<<rchannels<<std::endl;
	

	cv::Mat src_img = cv::imread("sample.jpg"); //refer to /Ultra-Light-Fast-Generic-Face-Detector-1MB/imgs/2.jpg
	printf("h:%d,w:%d\n",src_img.rows,src_img.cols);
    cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
    cv::resize(src_img, src_img, cv::Size(640, 640));

	std::cout<<"read file finish"<<std::endl;
	float* transformed_data = input->mutable_cpu_data();

	std::cout<<"begin transfor"<<std::endl;
	unsigned int volChl = rheight * rwidth;
	float pixel;
	for(int c=0; c<rchannels; ++c)
		for(unsigned int i=0; i<volChl; ++i)
		{
			pixel = static_cast<float>((float(src_img.data[i*3 + c])));
			transformed_data[c*volChl + i] = pixel;
		}
  
	std::cout<<"transform finish"<<std::endl;	
	caffe_net_test->Forward();
	std::cout<<"forward finish "<<std::endl;	
	
	caffe::Blob<float>* result_blob = caffe_net_test->output_blobs()[0];
    float prob_0 = caffe_net_test->blob_by_name("face_rpn_cls_prob_stride8")->data_at(0,0,0,0);
	int num_det = result_blob->height();

    std::cout<<"prob0 = "<<prob_0<<",index:"<<num_det<<std::endl;

	if(caffe_net_test != nullptr){
	//	delete caffe_net_test;
	//	caffe_net_test = nullptr;
	}
	
	
	//time test
	struct timeval t1,t2,t3;
    double timeuse,timeuse_warmup;
    gettimeofday(&t1,NULL);
	for(int i=0; i<23; ++i)
	{
		caffe_net_test->Forward();
		result_blob = caffe_net_test->output_blobs()[0];
		prob_0 = caffe_net_test->blob_by_name("face_rpn_cls_prob_stride8")->data_at(0,0,0,0);
		if(i == 2)
		{
			gettimeofday(&t2,NULL);
			timeuse_warmup = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 ;
		}
	}
	gettimeofday(&t3,NULL);
    timeuse = t3.tv_sec - t1.tv_sec + (t3.tv_usec - t1.tv_usec)/1000000.0;
	timeuse = (timeuse - timeuse_warmup)/ 20.0;
	printf("Warmup time:%f\n",timeuse_warmup);
    printf("Use Time:%f\n",timeuse);
	

	return 0;
}

