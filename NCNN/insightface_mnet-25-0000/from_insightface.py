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
sym = "mnet.25-symbol.json"
params =  "mnet.25-0000.params"
#model = mx.gluon.nn.SymbolBlock(outputs=mx.sym.load(sym), inputs=mx.sym.var('data'))
#model.load_params(params, ctx=mx.cpu())
#sym, arg_params, aux_params = mx.model.load_checkpoint('mnet.25', 0)
#model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
#symnet = mx.symbol.load(sym)
sym_, arg_params, aux_params = mx.model.load_checkpoint('mnet.25', 0)
model = mx.mod.Module(symbol=sym_, context=mx.cpu(), label_names=None)
model.bind(data_shapes=[('data', (1, 3, 300, 300))], for_training=False)
model.set_params(arg_params, aux_params, allow_missing=True)

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
img = cv2.resize(img,(300, 300)) 

image_data = img.transpose((2, 0, 1))
x = np.expand_dims(image_data, axis=0)
x = x.astype(np.float32)
'''
input_image = image.astype(dtype=np.float32)
input_image = input_image[:, :, :, np.newaxis]
input_image = input_image.transpose([3, 2, 0, 1])
'''
# Compile the model on Relay
# ---------------------------
# We should be familiar with the process right now.
input_tensor = "data"  
input_shape = x.shape
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
m.set_input("data", tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# set start time
start = timer()
for i in range(24):
    m.run()
    tvm_output = m.get_output(0)
    if i == 2:
        end = timer()
        timeuse_warmup = end - start
end = timer()
timeuse = end - start
print('warmup time' , timeuse_warmup)
print('time' , (timeuse - timeuse_warmup)/20.0)
print(tvm_output.shape)

'''
path_lib = "insightface_deploy_lib.so"
lib.export_library(path_lib)
with open("insightface_deploy_graph.json", "w") as fo:
    fo.write(graph)
with open("insightface_deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))
'''