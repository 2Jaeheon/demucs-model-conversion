import onnxruntime as ort
import sys

try:
    sess = ort.InferenceSession("../onnx-models/htdemucs_fp16.onnx", providers=['CPUExecutionProvider'])
    print("Inputs:")
    for i in sess.get_inputs():
        print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
    print("\nOutputs:")
    for o in sess.get_outputs():
        print(f"Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
except Exception as e:
    print(e)
