#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iterator>
#include <algorithm>
#include <sys/time.h>

/*
* opencv 读取的图像为HWC
* 根据模型需要转为 CHW
* 同时归一化
*/
void Mat2CHW(float *data, cv::Mat &frame, const int height, const int width, const int channel = 3)
{
    assert(data && !frame.empty());
    unsigned int volChl = height * width;
	for(int c=0; c<channel; ++c)
		for(unsigned int i=0; i<volChl; ++i)
		{
			data[c*volChl + i] = static_cast<float>((float(frame.data[i*3 + c])));
		}
}


int main()
{
    printf("starting run.\n");
    // tvm module for compiled functions
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("../lib/centerface_deploy_lib.so");
    printf("load lib.so ok.\n");

    // json graph
    std::ifstream json_in("centerface_deploy_graph.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();
    printf("load graph ok.\n");

    // parameters in binary
    std::ifstream params_in("centerface_deploy_param.params", std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();
    printf("load param ok.\n");

    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;

    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
    
    // load image data saved in binary
    //std::ifstream data_fin("cat.bin", std::ios::binary);
    //data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4);
    cv::Mat image, img_scale;
    image = cv::imread("sample.jpg");
    printf("h:%d,w:%d\n",image.rows,image.cols);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	int d_h = (int)(std::ceil(image.rows / 32.0) * 32);
	int d_w = (int)(std::ceil(image.cols / 32.0) * 32);
    cv::resize(image, img_scale, cv::Size(d_w, d_h));
	printf("---h:%d,w:%d\n",img_scale.rows,img_scale.cols);
	//cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));
    float *data = new float[img_scale.cols * img_scale.rows * 3];
    Mat2CHW(data, img_scale, img_scale.rows, img_scale.cols, 3);
	
	DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, img_scale.rows, img_scale.cols };
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

    //memcpy(x->data, &data, 3 * img_scale.rows * img_scale.cols * sizeof(float));
	TVMArrayCopyFromBytes(x, data, 3 * img_scale.rows * img_scale.cols * sizeof(float));
	delete[] data;
    printf("load data ok.\n");
	

    //time test
    int count = 0;
    
    
	int out_ndim = 4;
	int64_t out_shape[4] = {1, 2, 232, 400};
	
	// get the function from the module(load patameters)
	tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
	load_params(params_arr);

	// get the function from the module(run it)
	tvm::runtime::PackedFunc run = mod.GetFunction("run");
		
	DLTensor* y;
	TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

	// get the function from the module(get output data)
	tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
	//
	tvm::runtime::NDArray out;
    float* p_out;
	
    struct timeval t1,t2,t3;
    double timeuse,timeuse_warmup;
    gettimeofday(&t1,NULL);
    while(count < 9)
	{
		/* 
		// get the function from the module(set input data)
		tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
		set_input("input0", x);

		// get the function from the module(load patameters)
		tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
		load_params(params_arr);

		// get the function from the module(run it)
		tvm::runtime::PackedFunc run = mod.GetFunction("run");
		*/
		
		// get the function from the module(set input data)
	    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
	    set_input("input.1", x);
		run();
		//printf("run ok.\n");

		/* 
		//DLTensor* y;
		TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

		// get the function from the module(get output data)
		tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output"); 
		*/
		out = get_output(0);
		p_out = (float*)out.ToDLPack()->dl_tensor.data;
		
		get_output(1, y);

		count++;
		if(count == 1)
                {
                    gettimeofday(&t2,NULL);
                    timeuse_warmup = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
                }
	}
    gettimeofday(&t3,NULL);
    timeuse = t3.tv_sec - t1.tv_sec + (t3.tv_usec - t1.tv_usec)/1000000.0;
    timeuse = (timeuse - timeuse_warmup)/ 8.0;
    printf("Warmup time:%f\n",timeuse_warmup);
    printf("Use Time:%f\n",timeuse);
    printf("get function ok.\n");

    TVMArrayFree(x);
    TVMArrayFree(y);

    return 0;
}