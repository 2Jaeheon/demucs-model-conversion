# demucs.onnx

[Demucs v4 hybrid transformer](https://github.com/facebookresearch/demucs) 모델을 위한 ONNX 추론입니다. 이 모델은 음악 소스 분리를 위한 고성능, 고품질 PyTorch 신경망입니다.

다른 프로젝트 [demucs.cpp](https://github.com/sevagh/demucs.cpp)에서는 모든 신경망 레이어를 직접 작성했지만, 이 저장소에서는 [ONNXRuntime](https://github.com/microsoft/onnxruntime)을 사용합니다. ONNXRuntime은 다양한 하드웨어 플랫폼(CPU, GPU, 데스크톱 OS, 웹, 스마트폰 OS 등)을 네이티브로 지원하는 범용 고성능 신경망 추론 라이브러리입니다.

이 코드는 다른 프로젝트보다 훨씬 빠르게 수행되며, CUDA, GPU, WebGPU, WebNN 등 다양한 ONNX 실행 프로바이더를 활용할 수 있습니다.

## 아이디어 및 Python/PyTorch 구현

원본 Demucs v4는 STFT와 iSTFT(스펙트로그램 및 역 스펙트로그램)를 신경망 내부에서 수행하는데, 이러한 연산은 ONNX로 내보낼 수 없으므로 기본 내보내기는 보통 실패합니다:
```
torch.onnx.export(demucs_model, opset_version=17, ...)
```

[demucs-for-onnx](./demucs-for-onnx)에 demucs를 복사하고, `htdemucs.py`에서 stft/istft를 네트워크 자체 _외부_로 간단히 이동시켜서 핵심 모델이 스펙트로그램을 입력받고 반환하도록 했습니다:
```
# HTDemucs 클래스 메서드를 독립 함수로 이동
def standalone_spec(x, nfft=4096, hop_length=4096//4):
def standalone_magnitude(z, cac=True):
def standalone_ispec(z, length=None, scale=0, hop_length=4096//4):
def standalone_mask(z, m, cac=True, wiener_iters=0, training=False, end_iters=0):

class HTDemucs:
    # 입력 (mix, x) = 시간 도메인 파형과 complex-as-channels 스펙트로그램
    # 네트워크 자체에서 stft/istft 건너뛰기
    def forward(self, mix, x):
        ...
        return x, xt
```
`apply.py`에서, 모델 외부에서 호출하기 위해 spec/ispec/mag/mask 클래스 메서드의 독립 변형을 적용합니다:
```
# 이제 모델 자체에서 stft/istft를 제거했으므로
# 추론 전후에 이를 수행해야 합니다
with th.no_grad():
    training_length = int(model.segment * model.samplerate)
    # 이전에 모델에서 수행하던 패딩
    padded_padded_mix = F.pad(padded_mix, (0, training_length - padded_mix.shape[-1]))
    magspec = standalone_magnitude(standalone_spec(padded_padded_mix))
    out_x, out_xt = model(padded_mix, magspec)  # 핵심 모델 호출
    out = out_xt + out
    out = out[..., :valid_length]
```

## C++ 구현

C++에서는 demucs.cpp에서 STFT, iSTFT(complex-as-channels 포함) 및 패딩 함수를 가져와서 `src/model_inference.cpp`에서 ONNX demucs를 호출하기 전에 사용합니다:
```
// onnx를 사용한 핵심 demucs 추론 실행
void demucsonnx::model_inference(
    struct demucsonnx::demucs_model &model,
    struct demucsonnx::demucs_segment_buffers &buffers,
    struct demucsonnx::stft_buffers &stft_buf)
{
    // 먼저 스테레오 복소 스펙트로그램을 얻습니다
    demucsonnx::stft(stft_buf, buffers.padded_mix, buffers.z);

    // buffers.z를 input_tensors[1]에 복사하여 주파수 브랜치 입력 준비
    // x ('magnitude' 스펙트로그램 with complex-as-channels) 생성
    float *x_onnx_data = buffers.input_tensors[1].GetTensorMutableData<float>();
    ...

    // buffers.mix를 input_tensors[0]에 복사하여 시간 브랜치 입력 준비
    float *xt_onnx_data = buffers.input_tensors[0].GetTensorMutableData<float>();

    // 이제 stft가 있으므로, 핵심 demucs 추론 적용
    // (ONNX로 성공적으로 변환하기 위해 stft/istft를 제거한 곳)
    RunONNXInference(model, buffers);

    // 모델 실행
    model.sess->Run(
        demucsonnx::run_options,
        model.input_names_ptrs.data(),
        buffers.input_tensors.data(),
        buffers.input_tensors.size(),
        model.output_names_ptrs.data(),
        buffers.output_tensors.data(),
        model.output_names_ptrs.size()
    );

    std::cout << "ONNX inference completed." << std::endl;
```

## 설치 방법
먼저 모든 vendor 라이브러리(onnxruntime, Eigen 등)를 가져오기 위해 서브모듈과 함께 이 저장소를 클론합니다:
```
$ git clone --recurse-submodules https://github.com/sevagh/demucs.onnx
```

운영 체제에 맞는 표준 C++ 의존성을 설치합니다. 예: CMake, gcc, C++/g++, OpenBLAS (제 설명은 Pop!\_OS 22.04 기준입니다).

또한, 원하는 도구로 격리된 Python 환경을 설정하고([mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)를 선호합니다) `scripts/requirements.txt` 파일을 설치합니다:
```
$ mamba create --name demucsonnx python=3.12
$ mamba activate demucsonn
$ python -m pip install -r ./scripts/requirements.txt
```

### PyTorch 모델을 ONNX 및 ORT로 변환

Demucs PyTorch 모델을 ONNX로 변환:
```
$ python ./scripts/convert-pth-to-onnx.py ./demucs-onnx
...
Model successfully converted to ONNX format at onnx-models/htdemucs.onnx
```

4-source, 6-source, fine-tuned 모델을 변환할 수 있습니다. 그런 다음, ONNX를 ORT로 변환:
```
$ ./scripts/convert-model-to-ort.sh 
...
Converting optimized ONNX model /home/sevagh/repos/demucs.onnx/onnx-models/htdemucs.onnx to ORT format model /home/sevagh/repos/demucs.onnx/onnx-models/tmpmp673xjb.without_runtime_opt/htdemucs.ort
Converted 1/1 models successfully.
Generating config file from ORT format models with optimization style 'Runtime' and level 'all'
2024-11-11 08:10:05,695 ort_format_model.utils [INFO] - Created config in /home/sevagh/repos/demucs.onnx/onnx-models/htdemucs.required_operators_and_types.with_runtime_opt.config
```

### ONNXRuntime 빌드

[ort-builder](https://github.com/olilarkin/ort-builder) 전략을 사용하여, Demucs에 필요한 특정 타입과 연산자만 포함하는 최소한의 onnxruntime 라이브러리를 빌드합니다. Linux 빌드 스크립트(`./scripts/build-ort-linux.sh`)만 제공합니다.

그런 다음, 이 애플리케이션의 샘플 CLI 코드(`src_cli`에 있음)용 CMakeLists.txt 파일이 이 빌드된 onnxruntime 라이브러리에 링크됩니다.

### 이 C++ 코드 빌드

ONNXRuntime을 빌드한 후, CMake로 컴파일합니다 (최상위 Makefile에 정의된 편의 타겟을 통해):
```
$ make cli
$ ./build/build-cli/demucs ./onnx-models/htdemucs.ort ~/Music/unas.wav ./demucs-onnx-out
demucs.onnx Main driver program
Input samples: 2646000
Length in seconds: 60
Number of channels: 2
Running Demucs.onnx inference for: /home/sevagh/Music/unas.wav
shift offset is: 3062
ONNX inference completed.
(9.091%) Segment inference complete
...
ONNX inference completed.
(100.000%) Segment inference complete
Writing wav file "./demucs-onnx-out/target_0_drums.wav"
Encoder Status: 0
Writing wav file "./demucs-onnx-out/target_1_bass.wav"
Encoder Status: 0
Writing wav file "./demucs-onnx-out/target_2_other.wav"
Encoder Status: 0
Writing wav file "./demucs-onnx-out/target_3_vocals.wav"
Encoder Status: 0
```

## Credits
Based on [sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx) (MIT License)