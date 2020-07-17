import onnx
import numpy as np
import onnx.helper as helper
from onnx import shape_inference
import sys

# load model
net_name = sys.argv[1]
print(net_name)
onnx_model = onnx.load("./onnx_model/"+net_name)
graph = onnx_model.graph
node = graph.node
#print(node)

# rewrite the input tensor of graph
input_tensor = onnx.helper.make_tensor_value_info(name = "Placeholder_orig", elem_type = 1, shape = [1, 3, 513, 513])
graph.input.remove(graph.input[0])  # 删除旧节点
graph.input.insert(0, input_tensor)  # 插入新节点

print(graph.input)



value_info = graph.value_info

print("Before shape inference: \n")
#print(value_info)
print("------------------------------------------------------------")
print("After shape inference: \n")
inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
onnx.checker.check_model(onnx_model)
inferred_graph = inferred_onnx_model.graph
inferred_value_info = inferred_graph.value_info
#print(inferred_value_info)
#onnx.save(inferred_onnx_model,"./onnx_model/new_"+net_name)



# node_name = "MobilenetV2/expanded_conv/depthwise/depthwise"
# # node_index = 0
# # for i in range(len(node)):
# #     print(node[i].name)
# #     if node[i].name == node_name:
# #         node_index = i
# # print(node_index)
# # print(node[node_index])




