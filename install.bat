@echo off
REM Check Python version
python --version

REM Create Python virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   

REM Install CUDA-compatible PyTorch and dependencies
REM pip install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

REM Install project dependencies
pip install -r requirements.txt

REM Optional: Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

REM Optional: Verify installations
pip list

echo Installation complete. Activate the virtual environment with 'call venv\Scripts\activate' before running the application.
