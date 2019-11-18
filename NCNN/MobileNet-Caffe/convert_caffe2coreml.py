#convert caffe model to coreml model 
#from coremltools tutorial——
#https://apple.github.io/coremltools/generated/coremltools.converters.caffe.convert.html
import coremltools
coreml_model = coremltools.converters.caffe.convert(('mobilenet_v2.caffemodel','mobilenet_v2_deploy.prototxt'))
coreml_model.save('mobilenet_v2.mlmodel')
