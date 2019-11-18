import tvm
import tvm.relay as relay
import onnx
import numpy as np

from timeit import default_timer as timer
######################################################################
# Load pretrained CoreML model
# ----------------------------
#model_file = 'version-slim-320.onnx'
model_file = "version-RFB-320.onnx"
onnx_model = onnx.load(model_file)
print("load model ok.")

######################################################################
# Load a test image
# ------------------
import cv2
from matplotlib import pyplot as plt
import numpy as np
image_path = '/Ultra-Light-Fast-Generic-Face-Detector-1MB/imgs/2.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(320, 240))  #resize(width,height)
img_mean = np.array([127.,127.,127.])
img = (img - img_mean)/ 128.0
# after expand_dims, we have format NCHW
image_data = img.transpose((2, 0, 1))
x = np.expand_dims(image_data, axis=0)
x = x.astype(np.float32)
# Compile the model on Relay
# ---------------------------
# We should be familiar with the process right now.
input_tensor = "input"  # name of input data in model,the first layer name
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
for i in range(23):
    #complete a inference
    m.set_input("input", tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    m.run()
    tvm_output = m.get_output(0)
    if i == 2:
        end = timer()
        timeuse_warmup = end - start
end = timer()
timeuse = end - start
print('warmup time' , timeuse_warmup)
print('time' , (timeuse - timeuse_warmup)/20.0)
print(tvm_output)

path_lib = "ultra-light-fast-generic-face-detector-1MB_deploy_lib.so"
lib.export_library(path_lib)
with open("ultra-light-fast-generic-face-detector-1MB_deploy_graph.json", "w") as fo:
    fo.write(graph)
with open("ultra-light-fast-generic-face-detector-1MB_deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))