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
from wyoming.client import AsyncClient, AsyncTcpClient
from wyoming.event import Event
from wyoming.audio import AudioChunk, wav_to_chunks
from wyoming.asr import Transcribe, Transcript
import requests

# Import resample_audio
from resample_diagnostic import resample_audio

WHISPER_PORT = 8039 #10300


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
        
        # Transcribe using REST API
        transcriber = WhisperTranscriber()
        transcription = transcriber.transcribe_audio(resampled_temp_filename)
        
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

async def send_audio_file(audio_path: str, host: str = 'localhost', port: int = WHISPER_PORT):
    """
    Send an audio file to a Wyoming protocol service.
    
    Args:
        audio_path (str): Path to the WAV audio file to send
        host (str, optional): Hostname of the Wyoming service. Defaults to 'localhost'.
        port (int, optional): Port of the Wyoming service. Defaults to WHISPER_PORT.
    """
    try:
        # Open the WAV file
        with wave.open(audio_path, 'rb') as wav_file:
            # Inspect WAV file details
            rate = wav_file.getframerate()
            width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            
            logger.info(f"WAV File Details: rate={rate}, width={width}, channels={channels}")
            
            # Create Wyoming client
            async with AsyncTcpClient(host, port) as client:
                logger.info(f"Connected to Wyoming service at {host}:{port}")
                
                # Send audio start event
                start_event = Event(type="audio-start", data={
                    "rate": rate,
                    "width": width,
                    "channels": channels
                })
                logger.info(f"Sending audio-start event: {start_event}")
                await client.write_event(start_event)
                
                # Send audio chunks
                chunk_count = 0
                for chunk in wav_to_chunks(wav_file, samples_per_chunk=1000):
                    chunk_count += 1
                    logger.info(f"Chunk {chunk_count}: rate={chunk.rate}, width={chunk.width}, channels={chunk.channels}, audio_len={len(chunk.audio)}")
                    
                    # Create audio-chunk event
                    chunk_event = Event(
                        type="audio-chunk", 
                        data={
                            "rate": chunk.rate,
                            "width": chunk.width,
                            "channels": chunk.channels
                        },
                        payload=chunk.audio
                    )
                    await client.write_event(chunk_event)
                
                # Send audio stop event
                stop_event = Event(type="audio-stop")
                logger.info(f"Sending audio-stop event: {stop_event}")
                await client.write_event(stop_event)
                logger.info(f"Finished sending {chunk_count} audio chunks")
                
                # Wait for and print any responses
                while True:
                    event = await client.read_event()
                    if event is None:
                        break
                    logger.info(f"Received event: {event}")
    
    except Exception as e:
        logger.error(f"Error sending audio file: {e}")
        logger.error(traceback.format_exc())

async def main():
    """Main async entry point."""
    await send_audio_file('test_audio.wav')

class WhisperTranscriber:
    def __init__(self, host='localhost', port=8039):
        """
        Initialize Whisper transcriber
        
        :param host: Hostname of the Whisper service
        :param port: Port of the Whisper service
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.base_url = f"http://{host}:{port}"

    def transcribe_audio(self, audio_path: str):
        """
        Synchronous transcription using REST API
        
        :param audio_path: Path to WAV audio file
        :return: Transcription text
        """
        try:
            # Verify the audio file exists
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file not found: {audio_path}")
            
            self.logger.info(f"Transcribing audio file: {audio_path}")
            
            # Prepare the file for upload
            with open(audio_path, 'rb') as audio_file:
                files = {'file': ('audio.wav', audio_file, 'audio/wav')}
                data = {
                    'language': 'en'  # Set language to English
                }
                
                # Make POST request to transcription endpoint
                url = f"{self.base_url}/v1/audio/transcriptions"
                self.logger.info(f"Sending request to {url} with language=en")
                
                response = requests.post(url, files=files, data=data)
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                if isinstance(result, str):
                    # If response is just the text
                    transcription = result
                else:
                    # If response is a JSON object containing the text
                    transcription = result.get('text', '')
                
                self.logger.info(f"Received transcription: {transcription}")
                return transcription.strip()
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error during HTTP request: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {str(e)}")
            self.logger.error("Traceback:", exc_info=True)
            raise

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

@app.route('/')
def index():
    if request.method == 'POST':
        # Receive audio data
        audio_bytes = request.data
        
        # Convert audio bytes to numpy array
        try:
            audio_array, sr = convert_audio_to_float32(audio_bytes)
        except Exception as e:
            print(f"Error converting audio: {e}")
            return "Error processing audio", 400
        
        # Resample audio to 16kHz
        print(f"Original audio sample rate: {sr}Hz")
        resampled_audio = resample_audio(audio_array, sr)['scipy_poly']
        
        # Initialize transcriber
        transcriber = WhisperTranscriber()
        
        # Transcribe segments
        transcriptions = []
        transcription_thread = threading.Thread(
            target=run_async_transcription, 
            args=(transcriber, resampled_audio, 16000, transcriptions)
        )
        transcription_thread.start()
        transcription_thread.join()
        
        # Combine transcriptions
        full_transcription = " ".join(transcriptions)
        
        return render_template('simple.html', transcription=full_transcription)
    else:
        return render_template('simple.html', transcription="")

@app.route('/transcribe', methods=['POST'])
def transcribe():
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
        
        # Transcribe using REST API
        try:
            transcriber = WhisperTranscriber()
            transcription = transcriber.transcribe_audio(wav_filename)
            
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

@app.route('/check_whisper_health')
def check_whisper_health():
    """Check if the Whisper server is available"""
    try:
        url = f"http://localhost:8039/health"
        response = requests.get(url)
        response.raise_for_status()
        return jsonify({'status': 'ok'}), 200
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Whisper server health check failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 503

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5037)