import qai_hub as hub
import numpy as np

def save_qnn_output_as_raw(result_dict, raw_path='output.raw'):
    output = result_dict['output_0'][0]      # shape (1, 10)
    float_array = np.array(output, dtype=np.float32).reshape(-1)

    if len(float_array) != 10:
        raise ValueError('Expected 10 floats, got %d' % len(float_array))

    float_array.tofile(raw_path)
    print(f'Saved {len(float_array)} floats to raw file: {raw_path}')



# Compile
compile_job = hub.submit_compile_job(
    model="spoken_digit.onnx",
    device=hub.Device("Samsung Galaxy S23"),
    options="--target_runtime qnn_dlc",
    input_specs=dict(x=(1, 20, 35)),
)
assert isinstance(compile_job, hub.CompileJob)

# Profile 
profile_job = hub.submit_profile_job(
    model=compile_job.get_target_model(),
    device=hub.Device("Samsung Galaxy S23"),
)
assert isinstance(profile_job, hub.ProfileJob)

# Inputs
sample = np.fromfile("input.raw", dtype=np.float32).reshape(1, 20, 35)

# Inference
inference_job = hub.submit_inference_job(
    model=compile_job.get_target_model(),
    device=hub.Device("Samsung Galaxy S23"),
    inputs=dict(x=[sample]),
)
assert isinstance(inference_job, hub.InferenceJob)

# Save Output Data
output_tensors = inference_job.download_output_data()
print(output_tensors)
save_qnn_output_as_raw(output_tensors, 'output.raw')
