import onnx
from onnxconverter_common import float16
import sys
import os

if len(sys.argv) < 3:
    print("Usage: python convert_to_fp16.py <input_model> <output_model>")
    sys.exit(1)

input_model_path = sys.argv[1]
output_model_path = sys.argv[2]

if not os.path.exists(input_model_path):
    print(f"Error: Input model {input_model_path} does not exist.")
    sys.exit(1)

print(f"Loading model from {input_model_path}...")
model = onnx.load(input_model_path)

print("Converting to float16...")
model_fp16 = float16.convert_float_to_float16(model)

print(f"Saving model to {output_model_path}...")
onnx.save(model_fp16, output_model_path)
print("Done.")
