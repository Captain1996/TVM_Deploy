# insightface caffe model
## R50-models
 - [mxnet-model](https://pan.baidu.com/s/1WAkU9ZA_j-OmzO-sdk9whA)
 - [caffe-model](https://drive.google.com/drive/folders/1hA5x3jCYFdja3PXLl9EcmucipRmVAj3W) from [RetinaFace-Cpp](https://github.com/Charrin/RetinaFace-Cpp)

 ## problem
 - 经过R50-mxnet 转 ncnn模型，转换成功，但是读取模型发生字段错误
 - 经过mxnet转caffe模型，再转ncnn模型，转换成功，且读取模型和前传成功
 - 因为caffe原模型及caffe2ncnn原因，caffe和ncnn的模型输入都被固定为600*600
 
 ## reference
 - [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
 - [RetinaFace-Cpp](https://github.com/Charrin/RetinaFace-Cpp)
 - https://github.com/deepinsight/insightface/issues/669
