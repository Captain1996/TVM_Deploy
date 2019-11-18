######################################################################
# Utils for downloading and extracting zip files
# ---------------------------------------------
import os

def extract(path):
    import tarfile
    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError('Could not decompress the file: ' + path)


######################################################################
# Load pretrained TFLite model
# ---------------------------------------------
# we load mobilenet V1 TFLite model provided by Google
from tvm.contrib.download import download_testdata

model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"

# we download model tar file and extract, finally get mobilenet_v1_1.0_224.tflite
model_path = download_testdata(model_url, "mobilenet_v1_1.0_224.tgz", module=['tf', 'official'])
model_dir = os.path.dirname(model_path)
extract(model_path)

# now we have mobilenet_v1_1.0_224.tflite on disk and open it
tflite_model_file = os.path.join(model_dir, "mobilenet_v1_1.0_224.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()

# get TFLite model from buffer
import tflite.Model
tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Load a test image
# ---------------------------------------------
# A single cat dominates the examples!
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

#image_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
#image_path = download_testdata(image_url, 'cat.png', module='data')
image_path = 'cat.png'
resized_image = Image.open(image_path).resize((224, 224))
#plt.imshow(resized_image)
#plt.show()
image_data = np.asarray(resized_image).astype("float32")

# after expand_dims, we have format NHWC
image_data = np.expand_dims(image_data, axis=0)

# preprocess image as described here:
# https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
print('input', image_data.shape)
print(image_data[:, 0:10, 0:10, :])
resized_image = Image.open(image_path)
image_data_ = np.asarray(resized_image).astype("float32")
image_data_ = np.expand_dims(image_data_, axis=0)
print(image_data_[:,0:10,0:10,:])

######################################################################
# opencv 读图方式
# opencv read image
#-------------------------------------
#
# ---------------------------------------------
'''
import cv2
from matplotlib import pyplot as plt
import numpy as np

image_path = 'cat.png'
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(224,224))
img_ = np.zeros((224,224,3))
h = img.shape[0]
w = img.shape[1]
c = img.shape[2]
print(h,w,c)
for row in range(h):
    for col in range(w):
        for ch in range(c):
            pv = img[row,col,ch]
            img_[row,col,ch] = float(pv) * 2.0 / 255.0 -1
image_data = np.asarray(img_).astype("float32")

# after expand_dims, we have format NHWC
image_data = np.expand_dims(image_data, axis=0)

print('input', image_data.shape)
'''

######################################################################
# Compile the model with relay
# ---------------------------------------------

# TFLite input tensor name, shape and type
input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "float32"

# parse TFLite model and convert into Relay computation graph
from tvm import relay
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})

# target x86 CPU
target = "llvm"
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

######################################################################
# Execute on TVM
# ---------------------------------------------
import tvm
from tvm.contrib import graph_runtime as runtime

# create a runtime executor module
module = runtime.create(graph, lib, tvm.cpu())

# feed input data
module.set_input(input_tensor, tvm.nd.array(image_data))

# feed related params
module.set_input(**params)

# run
module.run()

# get output
tvm_output = module.get_output(0).asnumpy()

# save the graph, lib and params into separate files
from tvm.contrib import util

path_lib = "deploy_lib.so"
lib.export_library(path_lib)
with open("deploy_graph.json", "w") as fo:
    fo.write(graph)
with open("deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

######################################################################
# Display results
# ---------------------------------------------

# load label file
#label_file_url = ''.join(['https://raw.githubusercontent.com/',
#                          'tensorflow/tensorflow/master/tensorflow/lite/java/demo/',
#                          'app/src/main/assets/',
#                          'labels_mobilenet_quant_v1_224.txt'])
label_file = "labels_mobilenet_quant_v1_224.txt"
#label_path = download_testdata(label_file_url, label_file, module='data')

# list of 1001 classes
with open(label_file) as f:
    labels = f.readlines()

# convert result to 1D data
predictions = np.squeeze(tvm_output)

# get top 1 prediction
prediction = np.argmax(predictions)

# convert id to class name and show the result
print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])
print(predictions[prediction])