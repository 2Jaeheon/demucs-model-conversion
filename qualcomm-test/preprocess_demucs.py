"""
Demucs ONNX 모델용 전처리 스크립트

WAV/MP3 파일을 읽어 Demucs ONNX 모델의 두 가지 입력 형식으로 변환:
1. input.raw - 시간 도메인 오디오 (1, 2, 343980)
2. x.raw - STFT 결과 (1, 4, 2048, 336)
"""

import librosa
import numpy as np
import os
import sys

# Demucs 모델 상수 (C++ 코드와 동일)
SAMPLE_RATE = 44100
TIME_BRANCH_LEN = 343980  # 약 7.8초
FFT_WINDOW_SIZE = 4096
FFT_HOP_SIZE = 1024
FREQ_BRANCH_LEN = 336  # TIME_BRANCH_LEN / FFT_HOP_SIZE + 1

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

def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """오디오를 정확히 target_length 샘플로 패딩 또는 트림"""
    current_length = audio.shape[1]
    
    if current_length > target_length:
        # 트림
        return audio[:, :target_length]
    elif current_length < target_length:
        # 패딩
        padding = target_length - current_length
        return np.pad(audio, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    else:
        return audio

def compute_stft(audio: np.ndarray) -> np.ndarray:
    """
    STFT 계산하여 모델 입력 형식으로 변환
    
    입력: audio (2, time_samples)
    출력: x (1, 4, 2048, 336) - complex를 real/imag 채널로 분리
    """
    n_fft = FFT_WINDOW_SIZE
    hop_length = FFT_HOP_SIZE
    
    # 각 채널에 대해 STFT 수행
    stft_results = []
    for ch in range(2):
        # librosa STFT 사용
        stft = librosa.stft(
            audio[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        stft_results.append(stft)
    
    # Shape: (2, n_bins, n_frames) where n_bins = n_fft/2 + 1 = 2049
    stft_array = np.stack(stft_results, axis=0)
    
    # DC와 Nyquist 제외 (2049 -> 2048)
    # 모델은 2048 bins을 기대 (DC 제외)
    stft_array = stft_array[:, 1:2049, :]  # Skip DC, keep 2048 bins
    
    # 프레임 수 조정 (336 프레임)
    n_frames = stft_array.shape[2]
    if n_frames > FREQ_BRANCH_LEN:
        stft_array = stft_array[:, :, :FREQ_BRANCH_LEN]
    elif n_frames < FREQ_BRANCH_LEN:
        padding = FREQ_BRANCH_LEN - n_frames
        stft_array = np.pad(stft_array, ((0, 0), (0, 0), (0, padding)), mode='constant')
    
    # Complex를 real/imag 채널로 분리
    # (2, 2048, 336) complex -> (4, 2048, 336) float
    real_part = np.real(stft_array)
    imag_part = np.imag(stft_array)
    
    # 채널 순서: [L_real, L_imag, R_real, R_imag] = 4 channels
    x = np.zeros((4, 2048, FREQ_BRANCH_LEN), dtype=np.float32)
    x[0] = real_part[0]  # L real
    x[1] = imag_part[0]  # L imag
    x[2] = real_part[1]  # R real
    x[3] = imag_part[1]  # R imag
    
    # 배치 차원 추가: (1, 4, 2048, 336)
    return x.reshape(1, 4, 2048, FREQ_BRANCH_LEN).astype(np.float32)

def preprocess_audio(input_file: str, output_dir: str = '.'):
    """메인 전처리 함수"""
    print(f"Processing: {input_file}")
    
    # 1. 오디오 로드
    audio = load_audio(input_file)
    print(f"  Loaded audio shape: {audio.shape}, sample rate: {SAMPLE_RATE}Hz")
    
    # 2. 정확한 길이로 조정
    audio = pad_or_trim(audio, TIME_BRANCH_LEN)
    print(f"  Padded/trimmed to shape: {audio.shape}")
    
    # 3. 시간 도메인 입력 생성 (input)
    # Shape: (1, 2, 343980)
    input_tensor = audio.reshape(1, 2, TIME_BRANCH_LEN).astype(np.float32)
    input_path = os.path.join(output_dir, 'input.raw')
    input_tensor.tofile(input_path)
    print(f"  Saved time-domain input: {input_path}")
    print(f"    Shape: {input_tensor.shape}, Size: {os.path.getsize(input_path)} bytes")
    
    # 4. 주파수 도메인 입력 생성 (x)
    x_tensor = compute_stft(audio)
    x_path = os.path.join(output_dir, 'x.raw')
    x_tensor.tofile(x_path)
    print(f"  Saved frequency-domain input: {x_path}")
    print(f"    Shape: {x_tensor.shape}, Size: {os.path.getsize(x_path)} bytes")
    
    print("\nPreprocessing complete!")
    print(f"  input.raw: {input_tensor.shape} -> Model input 'input'")
    print(f"  x.raw: {x_tensor.shape} -> Model input 'x'")
    
    return input_tensor, x_tensor

def main():
    if len(sys.argv) < 2:
        print("Usage: python preprocess_demucs.py <audio_file> [output_dir]")
        print("Example: python preprocess_demucs.py ../day6-happy.mp3 .")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    if not os.path.isfile(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    preprocess_audio(input_file, output_dir)

if __name__ == '__main__':
    main()
