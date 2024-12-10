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

async def send_audio_file(audio_path: str, host: str = 'localhost', port: int = 10300):
    """
    Send an audio file to a Wyoming protocol service.
    
    Args:
        audio_path (str): Path to the WAV audio file to send
        host (str, optional): Hostname of the Wyoming service. Defaults to 'localhost'.
        port (int, optional): Port of the Wyoming service. Defaults to 10300.
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
    def __init__(self, host='localhost', port=10300):
        """
        Initialize Wyoming protocol Whisper transcriber
        
        :param host: Hostname of the Wyoming service
        :param port: Port of the Wyoming service
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)

    async def transcribe_audio(self, audio_path: str):
        """
        Transcribe audio using Wyoming protocol
        
        :param audio_path: Path to 16kHz WAV audio file
        :return: Transcription text
        """
        try:
            # Verify the audio file is 16kHz
            with wave.open(audio_path, 'rb') as wav_file:
                rate = wav_file.getframerate()
                width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                
                if rate != 16000:
                    raise ValueError(f"Audio must be 16kHz, got {rate}Hz")
                
                self.logger.info(f"WAV File Details: rate={rate}, width={width}, channels={channels}")
            
            # Create Wyoming client
            async with AsyncTcpClient(self.host, self.port) as client:
                self.logger.info(f"Connected to Wyoming service at {self.host}:{self.port}")
                
                # Send audio start event
                start_event = Event(type="audio-start", data={
                    "rate": rate,
                    "width": width,
                    "channels": channels
                })
                self.logger.info(f"Sending audio-start event: {start_event}")
                await client.write_event(start_event)
                
                # Open the WAV file
                with wave.open(audio_path, 'rb') as wav_file:
                    # Send audio chunks
                    chunk_count = 0
                    for chunk in wav_to_chunks(wav_file, samples_per_chunk=1000):
                        chunk_count += 1
                        self.logger.info(f"Chunk {chunk_count}: rate={chunk.rate}, width={chunk.width}, channels={chunk.channels}, audio_len={len(chunk.audio)}")
                        
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
                self.logger.info(f"Sending audio-stop event: {stop_event}")
                await client.write_event(stop_event)
                self.logger.info(f"Finished sending {chunk_count} audio chunks")
                
                # Collect transcription
                transcription = ""
                while True:
                    event = await client.read_event()
                    if event is None:
                        break
                    
                    self.logger.info(f"Received event: {event}")
                    
                    if event.type == "transcript":
                        transcription += event.data.get("text", "") + " "
                    elif event.type == "error":
                        self.logger.error(f"Transcription error: {event}")
                
                return transcription.strip()
        
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""

@app.route('/', methods=['GET', 'POST'])
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
        
        # Initialize transcriber
        transcriber = WhisperTranscriber()
        
        # Transcribe segments
        transcriptions = []
        transcription_thread = threading.Thread(
            target=run_async_transcription, 
            args=(transcriber, audio_array, sr, transcriptions)
        )
        transcription_thread.start()
        transcription_thread.join()
        
        # Combine transcriptions
        full_transcription = " ".join(transcriptions)
        
        return render_template('index.html', transcription=full_transcription)
    else:
        return render_template('index.html', transcription="")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Receive audio file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Log initial file information
        logging.info(f"TRANSCRIPTION REQUEST STARTED")
        logging.info(f"Received file: {file.filename}")
        logging.info(f"Content Type: {file.content_type}")
        logging.info(f"Content Length: {request.content_length}")
        
        # Inspect raw file content
        file.seek(0)
        raw_content = file.read()
        logging.info(f"Raw Content Length: {len(raw_content)} bytes")
        logging.info(f"First 50 bytes (hex): {raw_content[:50].hex()}")
        
        # Save raw input file
        temp_dir = os.path.join(os.path.dirname(__file__), '_temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename for raw file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_filepath = os.path.join(temp_dir, f'raw_input_{timestamp}.bin')
        
        # Save raw file content for inspection
        with open(raw_filepath, 'wb') as f:
            f.write(raw_content)
        
        logging.info(f"Saved raw input file to: {raw_filepath}")
        
        # Reset file stream
        file.seek(0)
        
        # Try to read as WAV
        try:
            # Create a BytesIO object from the file content
            import io
            wav_bytes_io = io.BytesIO(raw_content)
            
            # Attempt to read WAV file details
            with wave.open(wav_bytes_io, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                logging.info(f"ORIGINAL WAV FILE DETAILS: "
                             f"Channels: {n_channels}, "
                             f"Sample Width: {sample_width}, "
                             f"Framerate: {framerate}, "
                             f"Frames: {n_frames}")
                
                # Read audio data
                audio_data = wav_file.readframes(n_frames)
                
                # Conversion strategy based on sample width
                if sample_width == 1:
                    # 8-bit unsigned
                    audio_array = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
                    audio_array = (audio_array - 128) / 128.0
                elif sample_width == 2:
                    # 16-bit signed
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 3:
                    # 24-bit signed
                    audio_array = np.frombuffer(audio_data, dtype=np.int32, count=len(audio_data)//3) / (2**23 - 1)
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                # Mono conversion if multi-channel
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)
        
        except Exception as wav_error:
            # If WAV reading fails, try reading as raw audio
            logging.error(f"WAV file read error: {wav_error}")
            
            # Try to convert raw content to numpy array
            try:
                # Multiple conversion strategies
                conversion_strategies = [
                    (np.int16, 32768.0),    # Most common
                    (np.int32, 2147483648.0),  # 32-bit
                    (np.uint8, 256.0)       # 8-bit unsigned
                ]
                
                for dtype, divisor in conversion_strategies:
                    try:
                        audio_array = np.frombuffer(raw_content, dtype=dtype).astype(np.float32) / divisor
                        framerate = 16000  # Default sample rate
                        logging.info(f"Interpreted as {dtype} PCM. Assumed sample rate: {framerate}")
                        break
                    except Exception as strategy_error:
                        logging.warning(f"Conversion strategy {dtype} failed: {strategy_error}")
                else:
                    raise ValueError("No successful conversion strategy")
            
            except Exception as convert_error:
                logging.error(f"Failed to convert audio data: {convert_error}")
                return jsonify({"error": "Could not process audio data"}), 400
        
        # Diagnostic logging of audio array before saving
        logging.info("AUDIO ARRAY DIAGNOSTIC:")
        logging.info(f"Shape: {audio_array.shape}")
        logging.info(f"Dtype: {audio_array.dtype}")
        logging.info(f"Min: {audio_array.min()}")
        logging.info(f"Max: {audio_array.max()}")
        logging.info(f"Mean: {audio_array.mean()}")
        logging.info(f"Std Dev: {audio_array.std()}")
        
        # Save the processed audio
        try:
            # IMPORTANT: Pass the original framerate
            original_wav_path = save_wav_to_temp(audio_array, framerate, prefix='raw_audio', processing_stage='original')
            if original_wav_path is None:
                logging.error("Failed to save original WAV")
                return jsonify({"error": "Could not save audio file"}), 500
            logging.info(f"Saved original WAV to: {original_wav_path}")
            logging.warning(f"Original audio saved: {original_wav_path}")
            logging.warning(f"Original audio sample rate: {framerate}")
            logging.warning(f"Original audio shape: {audio_array.shape}")
            logging.warning(f"Original audio dtype: {audio_array.dtype}")
        except Exception as save_error:
            logging.error(f"Failed to save WAV: {save_error}")
            return jsonify({"error": "Could not save audio file"}), 500
        
        # Resample to 16kHz if necessary
        if framerate != 16000:
            # Use resample_audio function
            logging.warning(f"Resampling audio from {framerate} Hz to 16000 Hz")
            resampled_results = resample_audio(audio_array, framerate)
            
            # Choose one of the resampling methods (e.g., scipy_poly)
            resampled_audio = resampled_results['scipy_poly']
            
            # Save resampled audio at 16kHz
            original_wav_path = save_wav_to_temp(
                resampled_audio, 
                16000, 
                prefix='preprocessed_audio', 
                processing_stage='resampled',
                segment_number=0
            )
            
            logging.warning(f"Resampled audio saved: {original_wav_path}")
            logging.warning(f"Resampled audio sample rate: 16000")
            logging.warning(f"Resampled audio shape: {resampled_audio.shape}")
            logging.warning(f"Resampled audio dtype: {resampled_audio.dtype}")
        
        # Initialize transcriber
        transcriber = WhisperTranscriber()
        
        # Run transcription
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logging.info("Starting transcription")
            transcription = loop.run_until_complete(transcriber.transcribe_audio(original_wav_path))
            logging.info("Transcription complete")
            loop.close()
        except Exception as transcribe_error:
            logging.error(f"Transcription error: {transcribe_error}")
            return jsonify({"error": f"Transcription failed: {str(transcribe_error)}"}), 500
        
        return jsonify({
            "transcription": transcription,
            "language": "auto",
            "sample_rate": framerate,
            "original_wav_path": original_wav_path,
            "raw_input_path": raw_filepath
        })
    
    except Exception as e:
        logging.error(f"Unexpected error in transcription: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)