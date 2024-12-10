# My Whisper Speech-to-Text Application

## Prerequisites
- Python 3.8+
- CUDA-compatible NVIDIA GPU (recommended)
- Windows 10/11

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/my-whisper.git
cd my-whisper
```

2. Run Installation Script
```bash
install.bat
```

3. Activate Virtual Environment
```bash
call venv\Scripts\activate
```

## CUDA and PyTorch Installation

The installation script will:
- Install PyTorch with CUDA support using the official CUDA-specific index
- Use the latest compatible versions of torch, torchvision, and torchaudio
- Verify CUDA availability

### Troubleshooting CUDA Installation
- Ensure you have the latest NVIDIA GPU drivers
- Check CUDA Toolkit compatibility
- Verify PyTorch CUDA installation with `python -c "import torch; print(torch.cuda.is_available())"`

## Running the Application

### Backend Whisper Service
```bash
python whisper_service.py
```
- Runs on `http://127.0.0.1:5000`
- Debug mode is enabled

### Frontend
Open `index.html` in a modern web browser

### Testing Transcription
Use the test script to transcribe an audio file:
```bash
test.bat path\to\your\audio_file.wav
```
- Supports WAV files
- Requires Whisper service to be running

## Configuration

### Whisper Model
- Modify `MODEL_SIZE` in `whisper_service.py`
- Options: "tiny", "base", "small", "medium", "large-v2"

### Device Configuration
- Change `DEVICE` and `COMPUTE_TYPE` in `whisper_service.py`
  - `device`: "cuda" or "cpu"
  - `compute_type`: "float16", "int8_float16", "int8"

## Troubleshooting
- Ensure CUDA drivers are up-to-date
- Check GPU compatibility
- Verify PyTorch CUDA installation
- Confirm audio file is in WAV format

## License
MIT License
