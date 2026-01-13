"""
Qualcomm AI Hub를 사용한 Demucs ONNX 모델 테스트 스크립트 (청킹 버전)
NPU 메모리 제한을 우회하기 위해 오디오를 작은 청크로 나누어 처리합니다.

주요 기능:
1. 오디오를 오버랩 청크로 분할
2. 각 청크에 대해 개별 추론 실행
3. Overlap-add 방식으로 출력 병합
"""

import qai_hub as hub
import numpy as np
import librosa
import os
import sys
import argparse

# Demucs 모델 상수 (1초 segment 모델)
SAMPLE_RATE = 44100
ORIGINAL_TIME_BRANCH_LEN = 343980  # 원본 모델 입력 크기 (~7.8초)
FFT_WINDOW_SIZE = 4096
FFT_HOP_SIZE = 1024
FREQ_BINS = 2048
NB_SOURCES = 4  # drums, bass, other, vocals

# 청킹 파라미터 (1초 segment 모델 기준)
CHUNK_SIZE = 44100  # 1초 segment 모델의 입력 크기
OVERLAP = 11025     # 25% 오버랩 (~0.25초)
HOP_SIZE = CHUNK_SIZE - OVERLAP  # 33075

# 주파수 도메인 청크 크기 (모델 출력에서 확인된 값)
FREQ_CHUNK_LEN = 44  # 1초 segment 모델의 freq 프레임 수

# 모델 및 디바이스 설정
MODEL_PATH = "htdemucs_model/htdemucs_seg1.0s_fp16.onnx"
DEVICE_NAME = "Snapdragon 8 Elite QRD"


def load_audio(file_path: str) -> np.ndarray:
    """오디오 파일 로드 (스테레오, 44.1kHz)"""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)
    
    # 모노인 경우 스테레오로 변환
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    
    # 채널이 2개가 아닌 경우 처리
    if audio.shape[0] > 2:
        audio = audio[:2]
    
    return audio


def chunk_audio(audio: np.ndarray, chunk_size: int, hop_size: int) -> list:
    """
    오디오를 오버랩 청크로 분할
    
    Args:
        audio: (2, total_samples) 스테레오 오디오
        chunk_size: 각 청크의 샘플 수
        hop_size: 청크 간 간격 (hop_size = chunk_size - overlap)
    
    Returns:
        청크 리스트 [(start_idx, chunk_audio), ...]
    """
    total_samples = audio.shape[1]
    chunks = []
    
    start = 0
    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = audio[:, start:end]
        
        # 마지막 청크가 짧으면 패딩
        if chunk.shape[1] < chunk_size:
            padding = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, padding)), mode='constant')
        
        chunks.append((start, chunk))
        start += hop_size
        
        # 마지막 청크 처리
        if end >= total_samples:
            break
    
    print(f"  Split audio into {len(chunks)} chunks")
    print(f"  Chunk size: {chunk_size} samples ({chunk_size/SAMPLE_RATE:.2f}s)")
    print(f"  Overlap: {OVERLAP} samples ({OVERLAP/SAMPLE_RATE:.2f}s)")
    
    return chunks


def compute_stft_for_chunk(chunk: np.ndarray) -> np.ndarray:
    """
    청크에 대해 STFT 계산
    
    입력: chunk (2, chunk_samples)
    출력: x (1, 4, FREQ_BINS, freq_frames) float32
    """
    n_fft = FFT_WINDOW_SIZE
    hop_length = FFT_HOP_SIZE
    
    # 각 채널에 대해 STFT 수행
    stft_results = []
    for ch in range(2):
        stft = librosa.stft(
            chunk[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        stft_results.append(stft)
    
    # Shape: (2, n_bins, n_frames)
    stft_array = np.stack(stft_results, axis=0)
    
    # DC 제외 (2049 -> 2048)
    stft_array = stft_array[:, 1:FREQ_BINS+1, :]
    
    # 프레임 수 조정
    n_frames = stft_array.shape[2]
    if n_frames > FREQ_CHUNK_LEN:
        stft_array = stft_array[:, :, :FREQ_CHUNK_LEN]
    elif n_frames < FREQ_CHUNK_LEN:
        padding = FREQ_CHUNK_LEN - n_frames
        stft_array = np.pad(stft_array, ((0, 0), (0, 0), (0, padding)), mode='constant')
    
    # Complex를 real/imag로 분리
    real_part = np.real(stft_array)
    imag_part = np.imag(stft_array)
    
    x = np.zeros((4, FREQ_BINS, FREQ_CHUNK_LEN), dtype=np.float32)
    x[0] = real_part[0]  # L real
    x[1] = imag_part[0]  # L imag
    x[2] = real_part[1]  # R real
    x[3] = imag_part[1]  # R imag
    
    return x.reshape(1, 4, FREQ_BINS, FREQ_CHUNK_LEN).astype(np.float32)


def create_overlap_window(chunk_size: int, overlap: int) -> np.ndarray:
    """
    Overlap-add용 윈도우 생성 (fade-in/fade-out)
    """
    window = np.ones(chunk_size)
    
    # Fade-in at start
    fade_in = np.linspace(0, 1, overlap)
    window[:overlap] = fade_in
    
    # Fade-out at end
    fade_out = np.linspace(1, 0, overlap)
    window[-overlap:] = fade_out
    
    return window


def merge_outputs_with_overlap(outputs: list, original_length: int) -> np.ndarray:
    """
    청크 출력을 overlap-add 방식으로 병합
    
    Args:
        outputs: [(start_idx, output_array), ...] 
                 output_array shape: (4, 2, chunk_samples)
        original_length: 원본 오디오 길이
    
    Returns:
        merged: (4, 2, original_length)
    """
    # 결과 배열 초기화
    merged = np.zeros((NB_SOURCES, 2, original_length), dtype=np.float32)
    weights = np.zeros(original_length, dtype=np.float32)
    
    # 윈도우 생성
    window = create_overlap_window(CHUNK_SIZE, OVERLAP)
    
    for start_idx, output in outputs:
        end_idx = min(start_idx + CHUNK_SIZE, original_length)
        valid_len = end_idx - start_idx
        
        # 윈도우 적용
        for source in range(NB_SOURCES):
            for ch in range(2):
                merged[source, ch, start_idx:end_idx] += output[source, ch, :valid_len] * window[:valid_len]
        
        weights[start_idx:end_idx] += window[:valid_len]
    
    # 가중치로 정규화
    weights = np.maximum(weights, 1e-8)  # 0으로 나누기 방지
    for source in range(NB_SOURCES):
        for ch in range(2):
            merged[source, ch] /= weights
    
    return merged


def compile_model_for_chunk(device: hub.Device):
    """청크 크기에 맞게 모델 컴파일"""
    print("\n" + "="*60)
    print("STEP 1: Compiling model for chunked input")
    print("="*60)
    print(f"  Chunk time samples: {CHUNK_SIZE}")
    print(f"  Chunk freq frames: {FREQ_CHUNK_LEN}")
    
    model_path = MODEL_PATH
    
    compile_job = hub.submit_compile_job(
        model=model_path,
        device=device,
        options="--target_runtime qnn_dlc",
        input_specs=dict(
            input=((1, 2, CHUNK_SIZE), "float16"),
            x=((1, 4, FREQ_BINS, FREQ_CHUNK_LEN), "float16")
        ),
    )
    
    assert isinstance(compile_job, hub.CompileJob)
    print(f"Compile job submitted: {compile_job.job_id}")
    print("Waiting for compilation to complete...")
    
    target_model = compile_job.get_target_model()
    
    if target_model is None:
        print("❌ Compilation failed!")
        print(f"  Check job status at: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
        return None, None
    
    print(f"✅ Compilation successful!")
    print(f"  Target model: {target_model}")
    
    return compile_job, target_model


def profile_model(target_model, device: hub.Device):
    """모델 프로파일링"""
    print("\n" + "="*60)
    print("STEP 2: Profiling chunked model")
    print("="*60)
    
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
    )
    
    assert isinstance(profile_job, hub.ProfileJob)
    print(f"Profile job submitted: {profile_job.job_id}")
    print("Waiting for profiling to complete...")
    
    try:
        profile_result = profile_job.download_profile()
        print("\n✅ Profile Results:")
        print(f"  Inference time: {profile_result.get('inference_time', 'N/A')}")
        print(f"  Memory usage: {profile_result.get('peak_memory_bytes', 'N/A')}")
        return True
    except Exception as e:
        print(f"\n⚠️ Profile result: {e}")
        # 프로파일링 실패해도 추론 시도
        return False


def run_inference_for_chunk(target_model, device: hub.Device, 
                            input_tensor: np.ndarray, x_tensor: np.ndarray,
                            chunk_idx: int) -> np.ndarray:
    """단일 청크에 대한 추론 실행"""
    
    inference_job = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs=dict(
            input=[input_tensor.astype(np.float16)],
            x=[x_tensor.astype(np.float16)]
        ),
    )
    
    assert isinstance(inference_job, hub.InferenceJob)
    print(f"  Chunk {chunk_idx}: job {inference_job.job_id}", end=" ")
    
    output_data = inference_job.download_output_data()
    
    if output_data is None:
        print("❌ FAILED")
        return None
    
    print("✅")
    
    # 시간 도메인 출력 추출 (add_67)
    # Shape: (1, 4, 2, chunk_samples)
    if 'add_67' in output_data:
        output = np.array(output_data['add_67'][0])
        return output[0]  # (4, 2, chunk_samples)
    
    # 출력 키 확인
    print(f"    Available outputs: {list(output_data.keys())}")
    # 첫 번째 출력 사용
    first_key = list(output_data.keys())[0]
    output = np.array(output_data[first_key][0])
    return output[0] if output.ndim == 4 else output


def run_chunked_inference(target_model, device: hub.Device, 
                          chunks: list, original_length: int) -> np.ndarray:
    """모든 청크에 대해 추론 실행 및 병합"""
    print("\n" + "="*60)
    print(f"STEP 3: Running inference for {len(chunks)} chunks")
    print("="*60)
    
    outputs = []
    
    for i, (start_idx, chunk_audio) in enumerate(chunks):
        # 시간 도메인 입력
        input_tensor = chunk_audio.reshape(1, 2, CHUNK_SIZE).astype(np.float32)
        
        # 주파수 도메인 입력
        x_tensor = compute_stft_for_chunk(chunk_audio)
        
        # 추론
        output = run_inference_for_chunk(target_model, device, 
                                          input_tensor, x_tensor, i+1)
        
        if output is None:
            print(f"❌ Chunk {i+1} failed!")
            return None
        
        outputs.append((start_idx, output))
    
    print(f"\n✅ All {len(chunks)} chunks processed successfully!")
    
    # Overlap-add 병합
    print("\nMerging chunks with overlap-add...")
    merged = merge_outputs_with_overlap(outputs, original_length)
    print(f"  Merged output shape: {merged.shape}")
    
    return merged


def save_stems_as_raw(stems: np.ndarray, output_dir: str = "output"):
    """Stem을 raw 파일로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 전체 출력을 하나의 파일로 저장
    output_path = os.path.join(output_dir, "add_67.raw")
    # Shape: (4, 2, samples) -> (1, 4, 2, samples)
    output_with_batch = stems.reshape(1, *stems.shape).astype(np.float32)
    output_with_batch.tofile(output_path)
    print(f"  Saved: {output_path} (shape: {output_with_batch.shape})")


def main():
    parser = argparse.ArgumentParser(description="Chunked Demucs inference on Qualcomm AI Hub")
    parser.add_argument("audio_file", nargs="?", default="../day6-happy.mp3",
                        help="Input audio file")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile the model, skip inference")
    parser.add_argument("--profile-only", action="store_true",
                        help="Compile and profile only, skip inference")
    args = parser.parse_args()
    
    print("="*60)
    print("Demucs ONNX Model - Chunked Qualcomm AI Hub Test")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE_NAME}")
    print(f"Chunk size: {CHUNK_SIZE} samples ({CHUNK_SIZE/SAMPLE_RATE:.2f}s)")
    print(f"Overlap: {OVERLAP} samples ({OVERLAP/SAMPLE_RATE:.2f}s)")
    print(f"Memory reduction: ~{(1 - CHUNK_SIZE/ORIGINAL_TIME_BRANCH_LEN)*100:.0f}%")
    
    # 디바이스 설정
    device = hub.Device(DEVICE_NAME)
    
    # 1. 컴파일
    compile_job, target_model = compile_model_for_chunk(device)
    
    if target_model is None:
        sys.exit(1)
    
    if args.compile_only:
        print("\n✅ Compile-only mode completed!")
        return
    
    # 2. 프로파일링
    profile_success = profile_model(target_model, device)
    
    if args.profile_only:
        if profile_success:
            print("\n✅ Profile-only mode completed!")
        else:
            print("\n⚠️ Profile-only mode completed with warnings")
        return
    
    # 3. 오디오 로드 및 청킹
    print("\n" + "="*60)
    print("Loading and chunking audio")
    print("="*60)
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    audio = load_audio(args.audio_file)
    original_length = audio.shape[1]
    print(f"  Loaded: {args.audio_file}")
    print(f"  Shape: {audio.shape} ({original_length/SAMPLE_RATE:.2f}s)")
    
    chunks = chunk_audio(audio, CHUNK_SIZE, HOP_SIZE)
    
    # 4. 청크별 추론
    merged_output = run_chunked_inference(target_model, device, chunks, original_length)
    
    if merged_output is None:
        print("\n❌ Inference failed!")
        sys.exit(1)
    
    # 5. 출력 저장
    print("\n" + "="*60)
    print("STEP 4: Saving outputs")
    print("="*60)
    save_stems_as_raw(merged_output, "output")
    
    print("\n" + "="*60)
    print("All steps completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run postprocess_demucs.py to convert outputs to WAV:")
    print("     python postprocess_demucs.py output stems")


if __name__ == "__main__":
    main()
