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
import subprocess
from datetime import datetime
from wav_compare import compare_wav_files

from flask import Flask, render_template, request, jsonify
# from wyoming.client import AsyncClient, AsyncTcpClient
# from wyoming.event import Event
# from wyoming.audio import AudioChunk, wav_to_chunks
# from wyoming.asr import Transcribe, Transcript
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
print("--------------- start ---------------")

def insance_convert(filename="recorded_audio.wav"):

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device="cuda:0", # or mps for Mac devices
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    outputs = pipe(
        filename,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )

    print("--------------- outputs ---------------")
    return outputs
#



# Import resample_audio
from resample_diagnostic import resample_audio

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
from audio_utils import *
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add SocketIO initialization
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('test_response', {'message': 'Connected to Whisper Server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('transcribe')
def handle_transcribe(data):
    """
    Handle transcription via WebSocket
    
    Expects:
    - audio_data: Base64 encoded audio
    - sample_rate: Audio sample rate
    """

    print("--------------- socket ---------------")

    try:
        # Decode base64 audio data
        import base64
        audio_data = base64.b64decode(data['audio_data'])
        sample_rate = data.get('sample_rate', 16000)
        
        # Convert to numpy array and check audio quality
        audio_array = np.frombuffer(audio_data, np.int16)
        
        # Check if audio is meaningful (not just noise)
        if np.abs(audio_array).max() < 100:  # Adjust threshold as needed
            logging.warning("Audio seems to be noise or too quiet. Skipping transcription.")
            socketio.emit('transcription', {
                'text': '',
                'confidence': 0,
                'error': 'Low audio quality'
            })
            return
        
        # Resample audio to 16kHz
        print(f"Original audio sample rate: {sample_rate}Hz")
        resampled_audio = resample_audio(audio_array.astype(np.float32) / 32768.0, sample_rate)['scipy_poly']
        
        # Save 16kHz diagnostic audio file
        diagnostic_16khz_path = save_16khz_audio(resampled_audio)
        
        # Save resampled audio to temporary file
        resampled_temp_filename = os.path.join(temp_dir, f"preprocessed_audio_16000hz_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_seg00_resampled.wav")
        sf.write(resampled_temp_filename, resampled_audio, 16000)
        
        outputs = insance_convert(resampled_temp_filename)
        print("--------------- outputs ---------------")
        print(outputs)
        transcription = outputs["text"]
        
        # Log the transcription for debugging
        logging.info(f"Transcription result: {transcription}")
        
        # Emit transcription back to client
        socketio.emit('transcription', {
            'text': transcription.strip(),  # Remove any leading/trailing whitespace
            'confidence': 0.95,  # Example confidence
            'language': 'en'  # Example language
        })
        
    except Exception as e:
        print(f"Transcription error: {e}")
        emit('transcription_error', {'error': str(e)})

def save_16khz_audio(audio_data, filename_prefix='raw_audio_16khz'):
    """
    Save 16kHz audio file with a descriptive timestamp-based filename
    
    :param audio_data: Numpy array of audio data
    :param filename_prefix: Prefix for the filename
    :return: Full path to the saved audio file
    """
    # Generate a timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{filename_prefix}_{timestamp}_diagnostic.wav"
    
    # Ensure _temp directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    # Full path for the file
    full_path = os.path.join(temp_dir, filename)
    
    try:
        # Save the audio file at 16kHz
        sf.write(full_path, audio_data, 16000)
        logging.info(f"Saved 16kHz diagnostic audio file: {full_path}")
        return full_path
    except Exception as e:
        logging.error(f"Failed to save 16kHz audio file: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', transcription="")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    print("--------------- transcribe ---------------")
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400
        
        audio_file = request.files['audio']
        
        # Generate a unique filename
        filename = f'_temp/recording_{uuid.uuid4()}.webm'
        audio_file.save(filename)
        
        # Convert WebM to WAV at 16kHz
        wav_filename = f'_temp/recording_{uuid.uuid4()}.wav'
        
        # Use ffmpeg for conversion
        try:
            result = subprocess.run([
                'ffmpeg', 
                '-i', filename, 
                '-ar', '16000', 
                '-ac', '1', 
                wav_filename
            ], capture_output=True, text=True, check=True)
            
            # Log ffmpeg output for debugging
            logging.info(f"FFmpeg conversion stdout: {result.stdout}")
            logging.info(f"FFmpeg conversion stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Audio conversion error: {e}")
            logging.error(f"FFmpeg stderr: {e.stderr}")
            return jsonify({'error': 'Failed to convert audio'}), 500
        
        # Transcribe using Wyoming protocol
        try:
            outputs = insance_convert(wav_filename)
            print("--------------- outputs ---------------")
            print(outputs)
            transcription = outputs["text"]
            
            # Clean up temporary files
            os.unlink(filename)
            os.unlink(wav_filename)
            
            return jsonify({
                'transcription': transcription,
                'status': 'success'
            })
        
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return jsonify({'error': 'Transcription failed'}), 500
    
    except Exception as e:
        logging.error(f"Transcription route error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)