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
from flask import Flask, render_template, request, jsonify
from wyoming.client import AsyncClient, AsyncTcpClient
from wyoming.event import Event
from wyoming.audio import AudioChunk, wav_to_chunks
from wyoming.asr import Transcribe, Transcript
import requests

# Import resample_audio
from resample_diagnostic import resample_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
TEMP_DIR = os.path.join(os.getcwd(), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

WHISPER_PORT = 8039 #10300

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
        resampled_temp_filename = os.path.join(TEMP_DIR, f"preprocessed_audio_16000hz_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_seg00_resampled.wav")
        sf.write(resampled_temp_filename, resampled_audio, 16000)
        logging.info(f"Saved resampled audio to {resampled_temp_filename}")
        
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
    Save 16kHz audio file with a timestamp-based filename
    
    :param audio_data: Numpy array of audio data
    :param filename_prefix: Prefix for the filename (unused)
    :return: Full path to the saved audio file
    """
    try:
        # Create a timestamp (milliseconds)
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}.wav"
        
        # Save the audio file
        filepath = os.path.join(TEMP_DIR, filename)
        sf.write(filepath, audio_data, 16000)
        logging.info(f"Saved 16kHz audio to {filepath}")
        return filepath
        
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
    """Handle audio file upload and transcription"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({'error': 'Empty audio file'}), 400
            
        # Create timestamp for this transcription
        timestamp = int(time.time() * 1000)
        
        # Save original upload
        temp_upload = os.path.join(TEMP_DIR, f"{timestamp}_upload.webm")
        audio_file.save(temp_upload)
        logging.info(f"Saved original upload to {temp_upload}")
        
        # Convert to WAV using FFmpeg
        wav_filename = os.path.join(TEMP_DIR, f"{timestamp}.wav")
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', temp_upload,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                wav_filename
            ], check=True, capture_output=True, text=True)
            logging.info(f"Converted to WAV: {wav_filename}")
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg conversion failed: {e}")
            logging.error(f"FFmpeg stderr: {e.stderr}")
            return jsonify({'error': 'Failed to convert audio'}), 500
        
        # Transcribe using REST API
        try:
            transcriber = WhisperTranscriber()
            transcription = transcriber.transcribe_audio(wav_filename)
            
            # Save transcription to text file
            txt_filename = os.path.join(TEMP_DIR, f"{timestamp}.txt")
            logging.info(f"Saving transcription to {txt_filename}")
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(transcription)
            logging.info(f"Saved transcription to {txt_filename}")
            
            return jsonify({'transcription': transcription})
            
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return jsonify({'error': str(e)}), 500
            
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