import os
import shutil
import tempfile
from pathlib import Path
import torch
import torchaudio
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from demucs.htdemucs import standalone_spec, standalone_magnitude, standalone_ispec, standalone_mask

# Initialize FastAPI
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR.parent / "onnx-models/htdemucs_fp16.onnx"
OUTPUT_DIR = BASE_DIR / "results"
FRONTEND_DIR = BASE_DIR / "frontend"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ONNX Model
print(f"Loading ONNX Model from {MODEL_PATH}...")
sess = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
print("Model Loaded.")

def preprocess(wav):
    """
    Preprocess audio: Pad and compute spectrogram
    """
    # Wav is (2, T)
    # standalone_spec expects (B, C, T)
    wav = wav.unsqueeze(0)  # (1, 2, T)
    
    # Check if model handles dynamic shapes
    # Use 343980 as segment length.
    segment_length = 343980
    return segment_length

SEGMENT_LENGTH = 343980

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Load Audio
        # Load Audio using soundfile directly
        import soundfile as sf
        wav_np, sr = sf.read(tmp_path)
        wav = torch.from_numpy(wav_np).float()
        
        # Handle shape (Time, Channels) -> (Channels, Time)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.t()
        if sr != 44100:
            wav = torchaudio.functional.resample(wav, sr, 44100)
            sr = 44100
        
        # Ensure Stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]

        # Normalize
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        # Chunking Logic
        length = wav.shape[-1]
        
        # If length is small, pad
        if length < SEGMENT_LENGTH:
            wav = torch.nn.functional.pad(wav, (0, SEGMENT_LENGTH - length))
            length = SEGMENT_LENGTH
            
        pad_len = (SEGMENT_LENGTH - (length % SEGMENT_LENGTH)) % SEGMENT_LENGTH
        if pad_len > 0:
            wav = torch.nn.functional.pad(wav, (0, pad_len))
            
        # Reshape to chunks
        # (2, N * SEG) -> (N, 2, SEG)
        chunks = wav.unfold(1, SEGMENT_LENGTH, SEGMENT_LENGTH).permute(1, 0, 2) # (N, 2, SEG)
        
        out_chunks = []
        
        for i in range(chunks.shape[0]):
            chunk = chunks[i].unsqueeze(0) # (1, 2, SEG)
            
            # Preprocess
            spec = standalone_spec(chunk)
            magspec = standalone_magnitude(spec)
            
            # Prepare inputs
            # Ensure float16 for inputs
            chunk_np = chunk.numpy().astype(np.float16)
            magspec_np = magspec.numpy().astype(np.float16)
            
            # Run Inference
            # Inputs: input, x
            outputs = sess.run(None, {'input': chunk_np, 'x': magspec_np})
            
            output_spec = torch.from_numpy(outputs[0].astype(np.float32)) # (1, 4, 4, Fr, T)
            output_time = torch.from_numpy(outputs[1].astype(np.float32)) # (1, 4, 2, L)
            
            # Postprocess
            # standalone_mask handles permute and view_as_complex
            spec_complex = standalone_mask(None, output_spec, cac=True)
            
            # Inverse Spec
            wav_from_spec = standalone_ispec(spec_complex, length=SEGMENT_LENGTH)
            
            # Combine
            out_chunk = wav_from_spec + output_time
            out_chunks.append(out_chunk)
            
        # Concatenate
        out_wav = torch.cat(out_chunks, dim=-1) # (1, 4, 2, Total)
        
        # Remove padding
        out_wav = out_wav[..., :length - pad_len]
        
        # Save files
        file_id = os.path.splitext(os.path.basename(file.filename))[0]
        out_paths = {}
        sources = ['drums', 'bass', 'other', 'vocals']
        
        for idx, source in enumerate(sources):
            # out_wav is (1, 4, 2, L)
            stem = out_wav[0, idx, :, :]
            stem_path = OUTPUT_DIR / f"{file_id}_{source}.wav"
            # Use soundfile directly
            sf.write(str(stem_path), stem.t().numpy(), 44100)
            out_paths[source] = f"/results/{stem_path.name}"
            
        return out_paths

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return FileResponse(FRONTEND_DIR / "index.html")

# Serve static files
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")
