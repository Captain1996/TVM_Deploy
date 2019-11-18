# insightface caffe model
 ## models
 - [caffe-model](https://drive.google.com/drive/folders/1hA5x3jCYFdja3PXLl9EcmucipRmVAj3W) from [RetinaFace-Cpp](https://github.com/Charrin/RetinaFace-Cpp)

 ## mnet25-models
 - [caffemodel](https://github.com/Charrin/RetinaFace-Cpp/tree/master/convert_models/mnet)

 ## make
 - c++ -std=c++11 -O2 -fPIC -I/opt/caffe/include insightface.cpp -o insightface -L/opt/caffe/build/lib/ -lcaffe -lboost_system -lboost_thread -lglog -lopencv_highgui -lopencv_imgproc -lopencv_core

 ## env set
 - 使用BLVC/Caffe docker
 - 使用Install-opencv git Repositories
 
 ## problem
 - 经过R50-mxnet 转 ncnn模型，转换成功，但是读取模型发生字段错误
 - 经过mxnet转caffe模型，再转ncnn模型，转换成功，且读取模型和前传成功
 - 因为caffe原模型及caffe2ncnn原因，caffe和ncnn的模型输入都被固定为600*600
 
 ## reference
 - [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
 - [RetinaFace-Cpp](https://github.com/Charrin/RetinaFace-Cpp)
 - https://github.com/deepinsight/insightface/issues/669
