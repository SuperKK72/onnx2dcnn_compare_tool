import onnx
from onnx import shape_inference
import onnx.utils
import onnx.helper as helper
import onnxruntime as rt
import os
import time
import numpy as np
from PIL import Image
import cv2
import sys

network_config_file_path = './onnx_inference_config/' + sys.argv[1]
network_config_file = open(network_config_file_path)
network_config_parameters = []
for network_config_parameter in network_config_file:
    if len(network_config_parameter.strip()) != 0:
        # print(network_config_parameter)
        network_config_parameter = network_config_parameter[network_config_parameter.find('=')+1:].strip()
        if network_config_parameter[0] != '#':
            network_config_parameters.append(network_config_parameter)
network_config_file.close()

net_name    = network_config_parameters[0]
net_path    = network_config_parameters[1]

img_path    = network_config_parameters[2]
img_n       = int(network_config_parameters[3])
img_c       = int(network_config_parameters[4])
img_h       = int(network_config_parameters[5])
img_w       = int(network_config_parameters[6])
std         = float(network_config_parameters[7])
val_B       = float(network_config_parameters[8])
val_G       = float(network_config_parameters[9])
val_R       = float(network_config_parameters[10])
val_D       = float(network_config_parameters[11])

rlt_path    = network_config_parameters[12]

# model = onnx.load_model(net_path)
# graph = model.graph
# input = graph.input[0]
# print(input)


def run_onnxruntime():
    """Run test against onnxruntime backend."""

    # load model
    m = rt.InferenceSession(net_path)
    input_name = m.get_inputs()[0].name
    output_name = m.get_outputs()[0].name
    # output_name = "resnet_v2_50/conv1/Conv2D:0"
    # input_name = "Placeholder_orig"
    # output_name = "MobilenetV1/Predictions/Softmax:0"

    image = cv2.imread(img_path, -1)
    image = cv2.resize(image, (img_w, img_h))
    X = np.array(image).astype(np.float32)
    X[:, :, 0] = (X[:, :, 0] - val_B) * std
    X[:, :, 1] = (X[:, :, 1] - val_G) * std
    X[:, :, 2] = (X[:, :, 2] - val_R) * std
    if img_c == 4:
        X[:, :, 3] = (X[:, :, 3] - val_D) * std
    X = X.transpose((2, 0, 1))
    X = X.reshape((img_n, img_c, img_h, img_w))
    # print(X)

    start = time.time()
    results = m.run([output_name], {input_name: X})
    print(type(results))
    results=np.array(results)
    #print(type(results))
    #print(results.shape)
    #print(results)
    #results = results.flatten()
    # results = m.run(None, {input_name: X})
    end = time.time()

    # onnx_model = onnx.load(net_path)
    # graph = onnx_model.graph
    # prob_info = helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 24, 112, 112])
    # graph.output.insert(0, prob_info)
    # print(graph.output)

    print("net name: ", net_name)
    print("input size: ", X.shape)
    print("input name: ", input_name)
    print("output name: ", output_name)
    print("output shape: ", results[0][0].shape)
    print("time of runtime: ", end-start, "s")
    print("len of result: ", len(results[0][0].flatten()))
    print("max of result: ", max(results[0][0].flatten()))
    print("index of max result: ", np.argmax(results[0][0].flatten()))

    print("----> start to save!")
    f = open(rlt_path, 'w')
    for value in results[0][0].flatten():
        f.write(str(value))
        f.write('\n')
    f.close()
    print("----> results have been saved!")
    # return results

if __name__ == "__main__":
    run_onnxruntime()