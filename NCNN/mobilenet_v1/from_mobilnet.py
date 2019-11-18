import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import coremltools as cm
import numpy as np
from PIL import Image

from timeit import default_timer as timer
######################################################################
# Load pretrained CoreML model
# ----------------------------
model_file = 'mobilenet.mlmodel'
mlmodel = cm.models.MLModel(model_file)
print("load model ok.")

######################################################################
# Load a test image
# ------------------
# A single cat dominates the examples!
import cv2
from matplotlib import pyplot as plt
import numpy as np
image_path = '../cat.png'
img = cv2.imread(image_path)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#mobilenet <- input bgr data
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
            if ch == 0:   
                img_[row,col,ch] = (float(pv) - 103.94) * 0.017
            if ch == 1:
                img_[row,col,ch] = (float(pv) - 116.78) * 0.017
            if ch == 2:
                img_[row,col,ch] = (float(pv) - 123.68) * 0.017
image_data = np.asarray(img_).astype("float32")
# after expand_dims, we have format NHWC
image_data = image_data.transpose((2, 0, 1))
x = np.expand_dims(image_data, axis=0)
# Compile the model on Relay
# ---------------------------
# We should be familiar with the process right now.
input_tensor = "data"  # name of input data in model,the first layer name
input_shape = x.shape
shape_dict = {input_tensor:input_shape}
print("shape: ",shape_dict)

target = 'llvm'

# Parse CoreML model and convert into Relay computation graph
mod, params = relay.frontend.from_coreml(mlmodel, shape_dict)

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
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)

# set start time
start = timer()
for i in range(23):
    # set inputs
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)
    if i == 2:
        end = timer()
        timeuse_warmup = end - start
end = timer()
timeuse = end - start
print('warmup time' , timeuse_warmup)
print('time' , (timeuse - timeuse_warmup)/20.0)
top1 = np.argmax(tvm_output.asnumpy()[0])

print("top1 id:" ,top1)

# save the graph, lib and params into separate files
from tvm.contrib import util

path_lib = "mobilenet_deploy_lib.so"
lib.export_library(path_lib)
with open("mobilenet_deploy_graph.json", "w") as fo:
    fo.write(graph)
with open("mobile_deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))
#####################################################################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
synset_name = '../imagenet1000_clsid_to_human.txt'
with open(synset_name) as f:
    synset = eval(f.read())

# You should see the following result: Top-1 id 282 class name tiger cat
print('Top-1 id', top1, 'class name', synset[top1])
print('score ',tvm_output.asnumpy()[0][top1])

