import tvm
import tvm.relay as relay
import mxnet as mx
import numpy as np
import os

from timeit import default_timer as timer
######################################################################
# Load pretrained mxnet model
# ----------------------------
#sym = os.path.join("./", "./", "symbol_10_320_20L_5scales_v2_deploy.json")
#params = os.path.join("./", "./",  "train_10_320_20L_5scales_v2_iter_1000000.params")
sym = "symbol_10_320_20L_5scales_v2_deploy.json"
params =  "train_10_320_20L_5scales_v2_iter_1000000.params"
model = mx.gluon.nn.SymbolBlock(outputs=mx.sym.load(sym), inputs=mx.sym.var('data'))
model.load_params(params, ctx=mx.cpu())
print("load model ok.")

######################################################################
# Load a test image
# ------------------
import cv2
from matplotlib import pyplot as plt
import numpy as np
image_path = '/Ultra-Light-Fast-Generic-Face-Detector-1MB/imgs/2.jpg'
image = cv2.imread(image_path)
resize_scale = 1
shorter_side = min(image.shape[:2])
if shorter_side * resize_scale < 128:
    resize_scale = float(128) / shorter_side
image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
image_mean = np.array([127.5, 127.5, 127.5])
image = (image - image_mean)/ 127.5
input_image = image.astype(dtype=np.float32)
input_image = input_image[:, :, :, np.newaxis]
input_image = input_image.transpose([3, 2, 0, 1])
# Compile the model on Relay
# ---------------------------
# We should be familiar with the process right now.
input_tensor = "data"  
input_shape = input_image.shape
shape_dict = {input_tensor:input_shape}
print("shape: ",shape_dict)

target = 'llvm'
# Parse mxnet model and convert into Relay computation graph
mod, params = relay.frontend.from_mxnet(model, shape_dict)

with relay.build_config(opt_level=3):
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
#complete a inference
m.set_input("data", tvm.nd.array(input_image.astype(dtype)))
m.set_input(**params)
# set start time
start = timer()
for i in range(23):
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

'''
path_lib = "A-Light-and-Fast-Face-Detector-for-Edge-Devices_deploy_lib.so"
lib.export_library(path_lib)
with open("A-Light-and-Fast-Face-Detector-for-Edge-Devices_deploy_graph.json", "w") as fo:
    fo.write(graph)
with open("A-Light-and-Fast-Face-Detector-for-Edge-Devices_deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))
'''