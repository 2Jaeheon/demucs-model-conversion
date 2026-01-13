#!/usr/bin/env python
"""
Demucs 모델을 청킹용 작은 입력 크기로 ONNX로 변환하는 스크립트

기존 모델 (343980 samples, ~7.8초)은 NPU 메모리 한계로 인해
Qualcomm AI Hub에서 실행 불가능하므로, 더 작은 크기로 export합니다.

핵심: 모델의 segment 파라미터를 수정하여 실제로 작은 모델 생성
"""

import sys
import torch
from torch.nn import functional as F
import argparse
from pathlib import Path
from fractions import Fraction
from demucs.pretrained import get_model
from demucs.htdemucs import HTDemucs, standalone_spec, standalone_magnitude

DEMUCS_MODEL = "htdemucs"
DEMUCS_MODEL_6S = "htdemucs_6s"

SAMPLE_RATE = 44100
# 기본 segment: 더 작은 크기로 설정 (2초)
DEFAULT_SEGMENT_SECONDS = 2.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Demucs PyTorch models to ONNX with custom segment size')
    parser.add_argument("dest_dir", type=str, help="destination path for the converted model")
    parser.add_argument("--segment", type=float, default=DEFAULT_SEGMENT_SECONDS,
                        help=f"Segment duration in seconds (default: {DEFAULT_SEGMENT_SECONDS}s)")
    parser.add_argument("--six-source", default=False, action="store_true", 
                        help="convert 6s model (default: 4s)")
    parser.add_argument("--fp16", default=False, action="store_true",
                        help="Export in FP16 format")

    args = parser.parse_args()

    segment_seconds = args.segment
    segment_samples = int(segment_seconds * SAMPLE_RATE)
    
    print("="*60)
    print("Demucs ONNX Export - Small Segment Version")
    print("="*60)
    print(f"Segment: {segment_seconds}s ({segment_samples} samples)")
    print(f"Original: 7.8s (343980 samples)")
    print(f"Memory reduction: ~{(1 - segment_samples/343980)*100:.0f}%")
    print()

    dir_out = Path(args.dest_dir)
    dir_out.mkdir(parents=True, exist_ok=True)

    # Load the appropriate model
    model_name = DEMUCS_MODEL
    if args.six_source:
        model_name = DEMUCS_MODEL_6S
    
    print(f"Loading model: {model_name}")
    model = get_model(model_name)

    # Check if model is an instance of BagOfModels
    if isinstance(model, HTDemucs):
        core_model = model
    elif hasattr(model, 'models') and isinstance(model.models[0], HTDemucs):
        core_model = model.models[0]
    else:
        raise TypeError("Unsupported model type")

    # ★ 핵심: 모델의 segment 파라미터를 직접 수정
    original_segment = core_model.segment
    print(f"Original segment: {original_segment}s")
    
    # segment를 Fraction으로 설정 (Demucs 내부에서 사용하는 형식)
    core_model.segment = Fraction(segment_samples, SAMPLE_RATE)
    print(f"Modified segment: {core_model.segment}s ({float(core_model.segment):.4f}s)")
    
    # use_train_segment을 True로 하여 고정 크기 강제
    core_model.use_train_segment = True
    
    # 새로운 training_length 계산
    training_length = int(float(core_model.segment) * core_model.samplerate)
    print(f"Training length: {training_length} samples")

    # Prepare a dummy input tensor
    print(f"\nCreating dummy input with shape (1, 2, {segment_samples})")
    dummy_waveform = torch.randn(1, 2, segment_samples)
    
    # Pad to training_length if necessary
    if dummy_waveform.shape[-1] < training_length:
        padding = training_length - dummy_waveform.shape[-1]
        print(f"Padding waveform by {padding} samples")
        dummy_waveform = F.pad(dummy_waveform, (0, padding))
    
    print(f"Final waveform shape: {dummy_waveform.shape}")

    # Compute magnitude spectrogram
    magspec = standalone_magnitude(standalone_spec(dummy_waveform))
    print(f"Magnitude spectrogram shape: {magspec.shape}")

    dummy_input = (dummy_waveform, magspec)

    # Define output file name
    suffix = f"_seg{segment_seconds}s"
    if args.fp16:
        suffix += "_fp16"
    onnx_file_path = dir_out / f"{model_name}{suffix}.onnx"

    print(f"\nExporting to: {onnx_file_path}")

    # Export the core model to ONNX
    try:
        torch.onnx.export(
            core_model,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input', 'x'],
            output_names=['output', 'xt']  # freq output, time output
        )
        print(f"\n✅ Model successfully exported!")
        print(f"   Path: {onnx_file_path}")
        print(f"   Input (time): (1, 2, {dummy_waveform.shape[-1]})")
        print(f"   Input (freq): {magspec.shape}")
        
        # Convert to FP16 if requested
        if args.fp16:
            import onnx
            from onnxconverter_common import float16
            
            print("\nConverting to FP16...")
            model_fp32 = onnx.load(str(onnx_file_path))
            model_fp16 = float16.convert_float_to_float16(model_fp32)
            onnx.save(model_fp16, str(onnx_file_path))
            print(f"✅ FP16 conversion complete!")
            
    except Exception as e:
        print(f"\n❌ Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    print(f"Segment size: {segment_seconds}s ({segment_samples} samples)")
    print(f"Memory reduction: ~{(1 - segment_samples/343980)*100:.0f}%")
    print(f"\nTo test on Qualcomm AI Hub:")
    print(f"  1. Update run_qai_hub_chunked.py with new model path")
    print(f"  2. Update CHUNK_SIZE to {segment_samples}")
    print(f"  3. Run: python run_qai_hub_chunked.py")
