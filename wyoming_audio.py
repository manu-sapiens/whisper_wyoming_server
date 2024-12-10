"""Wyoming audio utilities for audio chunk conversion and processing."""
import io
import wave
import numpy as np
import soundfile as sf
from scipy import signal

from wyoming.audio import AudioChunk, wav_to_chunks, AudioChunkConverter

# Re-export the imported classes and functions for compatibility
__all__ = ['AudioChunk', 'wav_to_chunks', 'AudioChunkConverter']

def convert_audio_to_wyoming_chunk(audio_data, sample_rate=16000):
    """
    Convert numpy audio data to Wyoming AudioChunk.
    
    :param audio_data: Numpy array of audio data
    :param sample_rate: Target sample rate
    :return: Wyoming AudioChunk
    """
    # Ensure mono channel
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Normalize and convert to 16-bit PCM
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
    
    return AudioChunk(
        rate=sample_rate,
        width=2,  # 16-bit
        channels=1,
        audio=audio_bytes
    )

def create_test_wav_file(filename='test_audio.wav', duration=1, sample_rate=16000):
    """
    Create a test WAV file for Wyoming protocol testing.
    
    :param filename: Output filename
    :param duration: Duration in seconds
    :param sample_rate: Sample rate
    """
    # Generate a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Write to WAV file
    sf.write(filename, audio_data, sample_rate, subtype='PCM_16')
