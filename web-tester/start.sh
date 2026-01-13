#!/bin/bash
source .venv/bin/activate
echo "Starting Demucs ONNX Web Tester..."
echo "Open http://localhost:8000 in your browser."
uvicorn backend.main:app --port 8000 --reload
