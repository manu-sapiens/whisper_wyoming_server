import asyncio
import io
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import threading
import wave
import time
import os
import webrtcvad
import uuid
from datetime import datetime
from wav_compare import compare_wav_files

from flask import Flask, render_template, request, jsonify
from wyoming.client import AsyncClient, AsyncTcpClient
from wyoming.event import Event
from wyoming.audio import AudioChunk, wav_to_chunks
from wyoming.asr import Transcribe, Transcript

# Configure logging to be more verbose and write to a file
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all log levels
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_processing.log'),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)

# Verify _temp directory
temp_dir = os.path.join(os.path.dirname(__file__), '_temp')
try:
    os.makedirs(temp_dir, exist_ok=True)
    logging.info(f"Verified _temp directory: {temp_dir}")
    logging.info(f"_temp directory absolute path: {os.path.abspath(temp_dir)}")
    logging.info(f"_temp directory exists: {os.path.exists(temp_dir)}")
    logging.info(f"_temp directory is writable: {os.access(temp_dir, os.W_OK)}")
except Exception as e:
    logging.error(f"Failed to create _temp directory: {e}")

app = Flask(__name__)

def write_wav_file(filepath, audio_data, sample_rate):
    """
    Write audio data to a WAV file with a correct RIFF header
    
    :param filepath: Path to save the WAV file
    :param audio_data: NumPy array of audio data
    :param sample_rate: Sample rate of audio
    """
    # Ensure audio is 16-bit PCM
    if audio_data.dtype == np.float32:
        audio_data = (audio_data * 32767).astype(np.int16)
    elif audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16)
    
    # Ensure mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Prepare WAV file
    with wave.open(filepath, 'wb') as wav_file:
        # Set WAV parameters
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Write frames
        wav_file.writeframes(audio_data.tobytes())
    
    # Verify WAV file
    try:
        with wave.open(filepath, 'rb') as wav_file:
            logging.info(f"WAV File Created: {filepath}")
            logging.info(f"WAV Details: "
                         f"Channels: {wav_file.getnchannels()}, "
                         f"Sample Width: {wav_file.getsampwidth()}, "
                         f"Framerate: {wav_file.getframerate()}, "
                         f"Frames: {wav_file.getnframes()}")
    except Exception as e:
        logging.error(f"WAV file verification failed: {e}")
        raise

def manual_wav_header(filepath, audio_data, sample_rate):
    """
    Manually create WAV file with RIFF header
    
    :param filepath: Path to save the WAV file
    :param audio_data: NumPy array of audio data
    :param sample_rate: Sample rate of audio
    """
    # Ensure 16-bit PCM
    if audio_data.dtype == np.float32:
        audio_data = (audio_data * 32767).astype(np.int16)
    elif audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16)
    
    # Ensure mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Audio data parameters
    num_samples = len(audio_data)
    bytes_per_sample = 2
    num_channels = 1
    
    # WAV file header parameters
    file_size = 36 + num_samples * bytes_per_sample
    
    try:
        with open(filepath, 'wb') as f:
            # RIFF Header
            f.write(b'RIFF')
            f.write((file_size - 8).to_bytes(4, 'little'))
            f.write(b'WAVE')
            
            # fmt Subchunk
            f.write(b'fmt ')
            f.write((16).to_bytes(4, 'little'))  # Subchunk1Size
            f.write((1).to_bytes(2, 'little'))   # AudioFormat (PCM)
            f.write((num_channels).to_bytes(2, 'little'))
            f.write((sample_rate).to_bytes(4, 'little'))
            f.write((sample_rate * num_channels * bytes_per_sample).to_bytes(4, 'little'))
            f.write((num_channels * bytes_per_sample).to_bytes(2, 'little'))
            f.write((bytes_per_sample * 8).to_bytes(2, 'little'))
            
            # data Subchunk
            f.write(b'data')
            f.write((num_samples * bytes_per_sample).to_bytes(4, 'little'))
            
            # Write audio data
            f.write(audio_data.tobytes())
        
        logging.info(f"Manually created WAV: {filepath}")
        logging.info(f"Audio details: "
                     f"Samples: {num_samples}, "
                     f"Sample Rate: {sample_rate}, "
                     f"Channels: {num_channels}")
    
    except Exception as e:
        logging.error(f"Manual WAV creation failed: {e}")
        raise

def save_wav_to_temp(audio_array, sample_rate, prefix='original', suffix='.wav', segment_number=None, processing_stage=None):
    """
    Save numpy audio array to a WAV file in _temp directory
    
    :param audio_array: NumPy audio array
    :param sample_rate: Sample rate of audio
    :param prefix: Filename prefix
    :param suffix: File extension
    :param segment_number: Optional segment number for explicit tracking
    :param processing_stage: Optional stage of processing
    :return: Path to saved WAV file or None if audio is not meaningful
    """
    # Log stack trace to identify where 16kHz file is being generated
    import traceback
    if sample_rate == 16000:
        logging.warning(f"Attempting to save 16kHz audio file with prefix: {prefix}")
        logging.warning("Call stack:")
        for line in traceback.format_stack():
            logging.warning(line.strip())

    # Ensure _temp directory exists
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '_temp'))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Diagnostic logging
    logging.info(f"Attempting to save audio: {prefix}, stage: {processing_stage}, Sample Rate: {sample_rate}")
    
    # Ensure input is a numpy array
    audio_array = np.asarray(audio_array)
    
    # Deduplicate audio array if it looks like it's been duplicated
    if len(audio_array) % 2 == 0 and np.array_equal(audio_array[:len(audio_array)//2], audio_array[len(audio_array)//2:]):
        logging.warning("Detected duplicated audio array. Reducing to unique half.")
        audio_array = audio_array[:len(audio_array)//2]
    
    # Log initial audio array details
    logging.info(f"Input Audio Array Details:")
    logging.info(f"  Shape: {audio_array.shape}")
    logging.info(f"  Dtype: {audio_array.dtype}")
    logging.info(f"  Min: {audio_array.min()}")
    logging.info(f"  Max: {audio_array.max()}")
    logging.info(f"  Mean: {audio_array.mean()}")
    logging.info(f"  Std Dev: {audio_array.std()}")
    
    # Ensure mono
    if audio_array.ndim > 1:
        logging.info(f"Converting multi-channel audio to mono. Original shape: {audio_array.shape}")
        audio_array = audio_array.mean(axis=1)
    
    # Explicit conversion to float32
    audio_array = audio_array.astype(np.float32)
    
    # Normalize if not already in correct range
    if audio_array.min() < -1 or audio_array.max() > 1:
        logging.warning("Normalizing audio array")
        audio_data = audio_array / max(abs(audio_array.min()), abs(audio_array.max()))
    else:
        audio_data = audio_array
    
    # Scaling for 16-bit PCM
    audio_data_scaled = (audio_data * 32767).astype(np.int16)
    
    # Check if audio contains meaningful sound
    rms = np.sqrt(np.mean(audio_data**2))
    peak_amplitude = np.max(np.abs(audio_data))
    
    # Conditions for meaningful audio
    is_meaningful = (
        rms > 0.01 and  # RMS threshold to filter out very low-energy noise
        peak_amplitude > 0.05  # Peak amplitude threshold
    )
    
    if not is_meaningful:
        logging.info(f"Audio not saved: Low energy. RMS: {rms}, Peak: {peak_amplitude}")
        return None
    
    # Use precise timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Construct more descriptive filename
    rate_part = f"{sample_rate}hz"
    
    # Construct filename with explicit segment number formatting
    if segment_number is not None:
        segment_part = f"_seg{segment_number:02d}"  # This ensures seg00, seg01, seg02
    else:
        segment_part = ""
    
    stage_part = f"_{processing_stage}" if processing_stage else ""
    
    # Comprehensive filename structure
    filename = f"{prefix}_{rate_part}_{timestamp}{segment_part}{stage_part}{suffix}"
    filepath = os.path.join(temp_dir, filename)
    
    try:
        # Detailed logging before writing
        logging.info("Writing WAV file:")
        logging.info(f"  Filepath: {filepath}")
        logging.info(f"  Sample Rate: {sample_rate}")
        logging.info(f"  Audio Data Shape: {audio_data_scaled.shape}")
        logging.info(f"  Audio Data Dtype: {audio_data_scaled.dtype}")
        
        # Write file with soundfile
        sf.write(
            filepath, 
            audio_data_scaled, 
            sample_rate, 
            subtype='PCM_16'  # Explicitly set 16-bit PCM
        )
        
        # Verify file
        with sf.SoundFile(filepath, 'r') as sound_file:
            logging.info(f"Saved WAV file details:")
            logging.info(f"Filename: {filename}")
            logging.info(f"Channels: {sound_file.channels}")
            logging.info(f"File size: {os.path.getsize(filepath)} bytes")
            logging.info(f"RMS: {rms}, Peak Amplitude: {peak_amplitude}")
            
        return filepath
    
    except Exception as e:
        logging.error(f"Failed to save WAV file: {e}")
        raise

def detect_voice_activity(audio_data, sr=16000, frame_duration=30, aggressiveness=3):
    """
    Perform Voice Activity Detection using WebRTC VAD
    
    :param audio_data: Input audio numpy array
    :param sr: Sample rate (must be 8000, 16000, 32000, or 48000)
    :param frame_duration: Frame duration in milliseconds (30ms recommended)
    :param aggressiveness: VAD aggressiveness (0-3, higher is more aggressive)
    :return: List of voice segments (start, end indices)
    """
    # Logging setup
    logger = logging.getLogger(__name__)
    logger.info("VOICE ACTIVITY DETECTION START")
    
    # Input audio diagnostics
    logger.info(f"Input Audio Characteristics:")
    logger.info(f"Sample Rate: {sr} Hz")
    logger.info(f"Audio Array Shape: {audio_data.shape}")
    logger.info(f"Audio Array Dtype: {audio_data.dtype}")
    
    # Initialize VAD
    vad = webrtcvad.Vad(aggressiveness)
    
    # Calculate frame parameters
    frame_length = int(sr * frame_duration / 1000)  # Frame length in samples
    logger.info(f"Frame Duration: {frame_duration} ms")
    logger.info(f"Frame Length: {frame_length} samples")
    
    # Convert to 16-bit PCM for VAD
    pcm_data = (audio_data * 32767).astype(np.int16)
    
    # Detect voice activity
    voice_segments = []
    in_voice_segment = False
    segment_start = 0
    
    for i in range(0, len(pcm_data), frame_length):
        frame = pcm_data[i:i+frame_length]
        
        # Pad frame if it's too short
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')
        
        # Convert to bytes for VAD
        frame_bytes = frame.tobytes()
        
        try:
            is_speech = vad.is_speech(frame_bytes, sr)
        except Exception as e:
            logger.error(f"VAD error at frame {i}: {e}")
            continue
        
        # Voice segment tracking
        if is_speech and not in_voice_segment:
            segment_start = i
            in_voice_segment = True
            logger.info(f"Voice segment started at {segment_start/sr:.3f} seconds")
        
        elif not is_speech and in_voice_segment:
            segment_end = i
            voice_segments.append((segment_start, segment_end))
            logger.info(f"Voice segment ended: {segment_start/sr:.3f} - {segment_end/sr:.3f} seconds")
            in_voice_segment = False
    
    # Handle case where last segment is a voice segment
    if in_voice_segment:
        segment_end = len(pcm_data)
        voice_segments.append((segment_start, segment_end))
        logger.info(f"Final voice segment: {segment_start/sr:.3f} - {segment_end/sr:.3f} seconds")
    
    # Log detected voice segments
    logger.info(f"Total Voice Segments Detected: {len(voice_segments)}")
    for i, (start, end) in enumerate(voice_segments, 1):
        logger.info(f"Segment {i}: {start/sr:.3f} - {end/sr:.3f} seconds (Duration: {(end-start)/sr:.3f} seconds)")
    
    return voice_segments

def convert_audio_to_float32(audio_bytes):
    """
    Convert audio bytes to float32 numpy array
    
    :param audio_bytes: Input audio bytes
    :return: Numpy array of audio data
    """
    # Create a file-like object from bytes
    audio_io = io.BytesIO(audio_bytes)
    
    # Read audio file
    audio_data, sample_rate = sf.read(audio_io, dtype='float32')
    
    # Ensure mono channel
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    
    return audio_data, sample_rate

def run_async_transcription(transcriber, audio_data, sr, result_container):
    """
    Run async transcription in a separate thread
    
    :param transcriber: WhisperTranscriber instance
    :param audio_data: Input audio numpy array
    :param sr: Sample rate
    :param result_container: List to store transcription results
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    transcriptions = loop.run_until_complete(transcriber.transcribe_segments(audio_data, sr))
    result_container.extend(transcriptions)
    loop.close()

def diagnose_audio_data(audio_array, prefix='audio'):
    """
    Comprehensive audio data diagnostic function
    
    :param audio_array: NumPy audio array
    :param prefix: Logging prefix
    """
    # Basic statistics
    logging.info(f"[{prefix} Diagnosis] Shape: {audio_array.shape}")
    logging.info(f"[{prefix} Diagnosis] Dtype: {audio_array.dtype}")
    logging.info(f"[{prefix} Diagnosis] Min: {audio_array.min()}")
    logging.info(f"[{prefix} Diagnosis] Max: {audio_array.max()}")
    logging.info(f"[{prefix} Diagnosis] Mean: {audio_array.mean()}")
    logging.info(f"[{prefix} Diagnosis] Std Dev: {audio_array.std()}")
    
    # Check for constant values or extreme noise
    unique_values = np.unique(audio_array)
    logging.info(f"[{prefix} Diagnosis] Unique Values Count: {len(unique_values)}")
    logging.info(f"[{prefix} Diagnosis] First 10 Unique Values: {unique_values[:10]}")
    
    # Optional: Plot histogram if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.hist(audio_array, bins=50)
        plt.title(f'{prefix} Audio Data Distribution')
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            os.path.dirname(__file__), 
            '_temp', 
            f'{prefix}_audio_distribution.png'
        )
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"[{prefix} Diagnosis] Distribution plot saved: {plot_path}")
    except Exception as plot_error:
        logging.warning(f"Could not generate plot: {plot_error}")
