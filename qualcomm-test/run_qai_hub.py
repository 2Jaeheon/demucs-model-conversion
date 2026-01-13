"""
Qualcomm AI Hub를 사용한 Demucs ONNX 모델 테스트 스크립트

주요 기능:
1. Compile: ONNX → QNN DLC 변환
2. Profile: 성능 측정 (로딩 속도, 메모리 사용량 등)
3. Inference: 실제 추론 실행
"""

import qai_hub as hub
import numpy as np
import os
import sys

# Demucs 모델 상수
TIME_BRANCH_LEN = 343980
FREQ_BRANCH_LEN = 336
FREQ_BINS = 2048
NB_SOURCES = 4  # drums, bass, other, vocals

# 모델 및 디바이스 설정
# FP16 모델 사용 (메모리 절감을 위해)
MODEL_DIR = "htdemucs_model"
MODEL_PATH = "htdemucs_model/htdemucs_fp16.onnx"
DEVICE_NAME = "Snapdragon 8 Elite QRD"

# ONNX Runtime 모드 (True: CPU에서 실행, False: NPU에서 QNN DLC로 실행)
# NPU 메모리 부족 시 True로 설정
USE_ONNX_RUNTIME = False


def load_inputs(input_raw: str = "input.raw", x_raw: str = "x.raw"):
    """전처리된 .raw 파일에서 입력 로드"""
    # 시간 도메인 입력 (1, 2, 343980)
    input_tensor = np.fromfile(input_raw, dtype=np.float32).reshape(1, 2, TIME_BRANCH_LEN)
    
    # 주파수 도메인 입력 (1, 4, 2048, 336)
    x_tensor = np.fromfile(x_raw, dtype=np.float32).reshape(1, 4, FREQ_BINS, FREQ_BRANCH_LEN)
    
    print(f"Loaded inputs:")
    print(f"  input: {input_tensor.shape}")
    print(f"  x: {x_tensor.shape}")
    
    return input_tensor, x_tensor


def save_output_as_raw(result_dict: dict, output_dir: str = "."):
    """추론 결과를 .raw 파일로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    for output_name, output_data in result_dict.items():
        output_array = np.array(output_data[0], dtype=np.float32)
        output_path = os.path.join(output_dir, f"{output_name}.raw")
        output_array.tofile(output_path)
        print(f"  Saved {output_name}: shape {output_array.shape} -> {output_path}")
    
    return output_dir


def compile_model(model_path: str, device: hub.Device):
    """ONNX 모델을 QNN DLC로 컴파일 또는 ONNX Runtime 모드"""
    print("\n" + "="*60)
    
    if USE_ONNX_RUNTIME:
        print("STEP 1: Using ONNX Runtime mode (skip compilation)")
        print("="*60)
        print("  Mode: ONNX Runtime (CPU)")
        print(f"  Model: {model_path}")
        # ONNX Runtime 모드에서는 컴파일 없이 원본 ONNX 모델 사용
        return None, model_path
    
    print("STEP 1: Compiling model to QNN DLC")
    print("="*60)
    
    # External weights가 있는 경우 디렉토리로 업로드
    model_dir = os.path.dirname(os.path.abspath(model_path))
    model_name = os.path.basename(model_path)
    external_data_path = model_path + ".data"
    
    if os.path.exists(external_data_path):
        print(f"  Detected external weights: {os.path.basename(external_data_path)}")
        print(f"  Using directory format: {model_dir}")
        model_to_upload = model_dir
    else:
        model_to_upload = model_path
    
    compile_job = hub.submit_compile_job(
        model=model_to_upload,
        device=device,
        options="--target_runtime qnn_dlc",
        input_specs=dict(
            input=((1, 2, TIME_BRANCH_LEN), "float16"),
            x=((1, 4, FREQ_BINS, FREQ_BRANCH_LEN), "float16")
        ),
    )
    
    assert isinstance(compile_job, hub.CompileJob)
    print(f"Compile job submitted: {compile_job.job_id}")
    print("Waiting for compilation to complete...")
    
    target_model = compile_job.get_target_model()
    
    if target_model is None:
        print("❌ Compilation failed!")
        print(f"  Check job status at: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
        sys.exit(1)
    
    print(f"✅ Compilation successful!")
    print(f"  Target model: {target_model}")
    
    return compile_job, target_model


def profile_model(target_model, device: hub.Device):
    """모델 프로파일링 (성능 측정)"""
    print("\n" + "="*60)
    print("STEP 2: Profiling model")
    print("="*60)
    
    if USE_ONNX_RUNTIME:
        print("  Mode: ONNX Runtime (directly profiling ONNX)")
    
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
    )
    
    assert isinstance(profile_job, hub.ProfileJob)
    print(f"Profile job submitted: {profile_job.job_id}")
    print("Waiting for profiling to complete...")
    
    # 프로파일링 결과 출력
    try:
        profile_result = profile_job.download_profile()
        print("\nProfile Results:")
        print(f"  Inference time: {profile_result.get('inference_time', 'N/A')}")
        print(f"  Memory usage: {profile_result.get('peak_memory_bytes', 'N/A')}")
    except Exception as e:
        print(f"\n⚠️ Profile download failed: {e}")
        print("  Continuing to inference step...")
    
    return profile_job


def run_inference(target_model, device: hub.Device, input_tensor, x_tensor):
    """모델 추론 실행"""
    print("\n" + "="*60)
    print("STEP 3: Running inference")
    print("="*60)
    
    if USE_ONNX_RUNTIME:
        print("  Mode: ONNX Runtime (directly running ONNX)")
    
    inference_job = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs=dict(
            input=[input_tensor],
            x=[x_tensor]
        ),
    )
    
    assert isinstance(inference_job, hub.InferenceJob)
    print(f"Inference job submitted: {inference_job.job_id}")
    print("Waiting for inference to complete...")
    
    # 출력 다운로드
    output_data = inference_job.download_output_data()
    
    if output_data is None:
        print("❌ Inference failed! Output data is None.")
        print(f"  Check job status at: https://workbench.aihub.qualcomm.com/jobs/{inference_job.job_id}/")
        sys.exit(1)
    
    print("\n✅ Inference Results:")
    for name, data in output_data.items():
        arr = np.array(data[0])
        print(f"  {name}: shape {arr.shape}")
    
    return inference_job, output_data


def main():
    # 입력 파일 확인
    input_raw = "input.raw"
    x_raw = "x.raw"
    
    if not os.path.exists(input_raw) or not os.path.exists(x_raw):
        print("Error: Input files not found!")
        print("Please run preprocess_demucs.py first:")
        print("  python preprocess_demucs.py <audio_file>")
        sys.exit(1)
    
    # 모델 파일 확인
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    print("="*60)
    print("Demucs ONNX Model - Qualcomm AI Hub Test")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE_NAME}")
    
    # 디바이스 설정
    device = hub.Device(DEVICE_NAME)
    
    # 입력 로드
    input_tensor, x_tensor = load_inputs(input_raw, x_raw)
    
    # 1. 컴파일
    compile_job, target_model = compile_model(MODEL_PATH, device)
    
    # 2. 프로파일링
    profile_job = profile_model(target_model, device)
    
    # 3. 추론
    inference_job, output_data = run_inference(target_model, device, input_tensor, x_tensor)
    
    # 4. 출력 저장
    print("\n" + "="*60)
    print("STEP 4: Saving outputs")
    print("="*60)
    save_output_as_raw(output_data, "output")
    
    print("\n" + "="*60)
    print("All steps completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check output/ directory for raw output files")
    print("  2. Run postprocess_demucs.py to convert outputs to WAV")


if __name__ == "__main__":
    main()
