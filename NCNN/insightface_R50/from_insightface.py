import tvm
import tvm.relay as relay
import mxnet as mx
import numpy as np
import os

from timeit import default_timer as timer
######################################################################
# Load pretrained mxnet model
# ----------------------------
#sym = os.path.join("./", "./", "R50-symbol.json")
#params = os.path.join("./", "./",  "R50-0000.params")
sym = "R50-symbol.json"
params =  "R50-0000.params"
model = mx.gluon.nn.SymbolBlock(outputs=mx.sym.load(sym), inputs=mx.sym.var('data'))
model.load_params(params, ctx=mx.cpu())
print("load model ok.")

######################################################################
# Load a test image
# ------------------
import cv2
from matplotlib import pyplot as plt
import numpy as np
image_path = 'sample.jpg'
img = cv2.imread(image_path)
resize_scale = 1
scales = [1024, 1980]
im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
#im_scale = 1.0
im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)
print('im_scale', im_scale)
if im_scale!=1.0:
    im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
im = im.astype(np.float32)
im_info = [im.shape[0], im.shape[1]]
im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
pixel_means=[0.0, 0.0, 0.0]
pixel_stds=[1.0, 1.0, 1.0]
pixel_scale = 1.0
for i in range(3):
    im_tensor[0, i, :, :] = (im[:, :, 2 - i]/pixel_scale - pixel_means[2 - i])/pixel_stds[2-i]

'''
input_image = image.astype(dtype=np.float32)
input_image = input_image[:, :, :, np.newaxis]
input_image = input_image.transpose([3, 2, 0, 1])
'''
# Compile the model on Relay
# ---------------------------
# We should be familiar with the process right now.
input_tensor = "data"  
input_shape = im_tensor.shape
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
m.set_input("data", tvm.nd.array(im_tensor.astype(dtype)))
m.set_input(**params)
# set start time
start = timer()
for i in range(9):
    m.run()
    tvm_output = m.get_output(0)
    if i == 0:
        end = timer()
        timeuse_warmup = end - start
end = timer()
timeuse = end - start
print('warmup time' , timeuse_warmup)
print('time' , (timeuse - timeuse_warmup)/8.0)
print(tvm_output.shape)

'''
path_lib = "insightface_deploy_lib.so"
lib.export_library(path_lib)
with open("insightface_deploy_graph.json", "w") as fo:
    fo.write(graph)
with open("insightface_deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))
'''