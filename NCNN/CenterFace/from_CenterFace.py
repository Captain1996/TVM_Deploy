import tvm
import tvm.relay as relay
import onnx
import numpy as np

from timeit import default_timer as timer
######################################################################
# Load pretrained Onnx model
# ----------------------------
model_file = "centerface.onnx"
onnx_model = onnx.load(model_file)
print("load model ok.")

######################################################################
# Load a test image
# ------------------
import cv2
from matplotlib import pyplot as plt
import numpy as np
image_path = 'sample.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#source code indicates that image is required resize to a format divided by 32
#The inferior code is self-defining according the image size.
d_h = (int)(img.shape[0] / 32.0 + 1)* 32
d_w = (int)(img.shape[1] / 32.0)* 32
img = cv2.resize(img,(d_w, d_h))  #resize(width,height)
#source code use function named cv::dnn::blobFromImage()
#it returns 4-dimensional Mat with NCHW dimensions order.
# thus convert HWC to CHW
#
# after expand_dims, we have format NCHW
image_data = img.transpose((2, 0, 1))
x = np.expand_dims(image_data, axis=0)
x = x.astype(np.float32)
# Compile the model on Relay
# ---------------------------
# We should be familiar with the process right now.
input_tensor = "input.1"  # name of input data in model,the first layer name
input_shape = x.shape
shape_dict = {input_tensor:input_shape}
print("shape: ",shape_dict)

target = 'llvm'
# Parse onnx model and convert into Relay computation graph
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with relay.build_config(opt_level=1):
    graph, lib, params = relay.build(mod,
                                         target,
                                         params=params)

######################################################################
# Execute on TVM
# -------------------
# The process is no different from other example
from tvm.contrib import graph_runtime

ctx = tvm.cpu(0)
m = graph_runtime.create(graph, lib, ctx)
dtype = 'float32'
# set start time
start = timer()
for i in range(9):
    #complete a inference
    m.set_input("input.1", tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    m.run()
    heatmap = m.get_output(0)
    scale = m.get_output(1)
    offset = m.get_output(2)
    landmarks = m.get_output(3)
    if i == 0:
        end = timer()
        timeuse_warmup = end - start
end = timer()
timeuse = end - start
print('warmup time' , timeuse_warmup)
print('time' , (timeuse - timeuse_warmup)/8.0)
print(heatmap.shape)
print(scale.shape)
print(offset.shape)
print(landmarks.shape)

path_lib = "centerface_deploy_lib.so"
lib.export_library(path_lib)
with open("centerface_deploy_graph.json", "w") as fo:
    fo.write(graph)
with open("centerface_deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))