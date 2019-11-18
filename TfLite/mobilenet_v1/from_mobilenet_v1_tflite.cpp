#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iterator>
#include <algorithm>
#include <sys/time.h>

void NORMAL(float *data, cv::Mat &frame)
{
    assert(data && !frame.empty());
    unsigned int volChl = 224 * 3;
    for(int i=0;i < 224*224*3;++i)
    {
	data[i] = static_cast<float>((float(frame.data[i])*2.0 / 255.0) -1.0);
    }
}


int main()
{
    printf("starting run.\n");
    // tvm module for compiled functions
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("lib/deploy_lib.so");//"deploy_lib.so");
    printf("load lib.so ok.\n");

    // json graph
    std::ifstream json_in("deploy_graph.json", std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();
    printf("load graph ok.\n");

    // parameters in binary
    std::ifstream params_in("deploy_param.params", std::ios::binary);
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
    
    DLTensor* x;
    int in_ndim = 4;
    int64_t in_shape[4] = {1, 3, 224, 224 };
    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
    // load image data saved in binary
    //std::ifstream data_fin("cat.bin", std::ios::binary);
    //data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4);
    cv::Mat image, input , frame;
    image = cv::imread("cat.png");
    cv::cvtColor(image, frame, cv::COLOR_BGR2RGB);
    cv::resize(frame, input, cv::Size(224, 224));
    float data[224 * 224 * 3];
    NORMAL(data, input);

    memcpy(x->data, &data, 3 * 224 * 224 * sizeof(float));
    printf("load data ok.\n");

    //time test
    int count = 0;
    
    DLTensor* y;
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);
    while(count < 100){
    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("input", x);

    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();
    
    //printf("run ok.\n");

    //DLTensor* y;
    int out_ndim = 2;
    int64_t out_shape[2] = {1, 1001, };
    TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    get_output(0, y);

    count++;}
    gettimeofday(&t2,NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    printf("Use Time:%f\n",timeuse);
    printf("get function ok.\n");

    // get the maximum position in output vector
    auto y_iter = static_cast<float*>(y->data);
    auto max_iter = std::max_element(y_iter, y_iter + 1001);
    auto max_index = std::distance(y_iter, max_iter);
    std::cout << "The maximum position in output vector is: " << max_index  << " , score: " << y_iter[max_index] << std::endl;

    TVMArrayFree(x);
    TVMArrayFree(y);

    return 0;
}