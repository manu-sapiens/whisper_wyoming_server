"""
Analyze WAV files to diagnose recording and conversion issues.
"""

import wave
import numpy as np
import os
import sys
import struct

def analyze_wav_file(file_path):
    """
    Comprehensive WAV file analysis.
    """
    print(f"\nAnalyzing WAV file: {file_path}")
    
    # Check file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return False
    
    # Read raw file contents
    with open(file_path, 'rb') as f:
        raw_bytes = f.read()
        print("First 50 bytes (hex):", raw_bytes[:50].hex())
    
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # Basic WAV parameters
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read audio data
            audio_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif sample_width == 1:
                audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                audio_array = (audio_array.astype(np.float32) - 128) * 256
            else:
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Audio analysis
            print(f"WAV File Details:")
            print(f"  Channels: {n_channels}")
            print(f"  Sample Width: {sample_width} bytes")
            print(f"  Frame Rate: {framerate} Hz")
            print(f"  Total Frames: {n_frames}")
            print(f"  Duration: {n_frames / framerate:.2f} seconds")
            
            # Audio characteristics
            print(f"\nAudio Characteristics:")
            print(f"  Min Amplitude: {audio_array.min()}")
            print(f"  Max Amplitude: {audio_array.max()}")
            print(f"  Mean Amplitude: {audio_array.mean()}")
            print(f"  RMS: {np.sqrt(np.mean(audio_array**2))}")
            
            return True
    
    except Exception as e:
        print(f"Error analyzing WAV file: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_wav.py <wav_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    analyze_wav_file(file_path)

if __name__ == '__main__':
    main()
