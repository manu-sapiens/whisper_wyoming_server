import numpy as np
import soundfile as sf
from scipy import signal
import librosa
import logging

def resample_audio(audio_data, original_sr):
    """
    Resample audio data to 16kHz using multiple methods and compare results
    
    :param audio_data: Input audio numpy array
    :param orig_sr: Original sample rate
    :param target_sr: Target sample rate
    :return: Dictionary of resampled audio arrays
    """
    # Ensure input is a numpy array
    audio_data = np.asarray(audio_data)
    
    # Normalize input audio to [-1, 1] range
    print("Pre-Normalization Input Audio:")
    print(f"  Min: {audio_data.min()}")
    print(f"  Max: {audio_data.max()}")
    print(f"  Mean: {audio_data.mean()}")
    print(f"  Std Dev: {audio_data.std()}")
    
    # Normalize if not already in correct range
    if audio_data.min() < -1 or audio_data.max() > 1:
        print("Normalizing input audio...")
        audio_data = audio_data / max(abs(audio_data.min()), abs(audio_data.max()))
    
    # Compute ratio
    ratio = target_sr / orig_sr

    # Method 1: scipy.signal.resample
    scipy_resampled = signal.resample(
        audio_data, 
        num=int(len(audio_data) * ratio)
    )
    
    # Method 2: scipy.signal.resample_poly (potentially higher quality)
    scipy_poly_resampled = signal.resample_poly(
        audio_data, 
        up=target_sr, 
        down=orig_sr
    )
    
    # Log resampled audio details
    print("\nResampled Audio Details:")
    
    print("scipy_basic:")
    print(f"  Shape: {scipy_resampled.shape}")
    print(f"  Min: {scipy_resampled.min()}")
    print(f"  Max: {scipy_resampled.max()}")
    print(f"  Mean: {scipy_resampled.mean()}")
    print(f"  Std Dev: {scipy_resampled.std()}")
    
    print("\nscipy_poly:")
    print(f"  Shape: {scipy_poly_resampled.shape}")
    print(f"  Min: {scipy_poly_resampled.min()}")
    print(f"  Max: {scipy_poly_resampled.max()}")
    print(f"  Mean: {scipy_poly_resampled.mean()}")
    print(f"  Std Dev: {scipy_poly_resampled.std()}")

    return {
        'scipy_basic': scipy_resampled,
        'scipy_poly': scipy_poly_resampled
    }
