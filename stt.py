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

    async def transcribe_audio(self, audio_data, sample_rate=16000):
        """
        Transcribe audio using Wyoming protocol
        
        :param audio_data: NumPy array of audio data
        :param sample_rate: Sample rate of audio
        :return: Transcription text
        """
        # Resample audio if sample rate is not 16000 Hz
        if sample_rate != 16000:
            self.logger.info(f"Resampling audio from {sample_rate} Hz to 16000 Hz")
            
            # Use resample_audio function
            resampled_results = resample_audio(audio_data, sample_rate)
            
            # Choose one of the resampling methods (e.g., scipy_poly)
            audio_data = resampled_results['scipy_poly']
            sample_rate = 16000
            
            # Log and save resampled audio
            self.logger.info(f"Original audio length: {len(audio_data)}")
            self.logger.info(f"Resampled audio length: {len(audio_data)}")
            
            # Save resampled audio
            save_wav_to_temp(
                audio_data, 
                sample_rate, 
                prefix='preprocessed_audio', 
                processing_stage='resampled',
                segment_number=0
            )
        
        try:
            # Establish connection to Wyoming service
            async with AsyncTcpClient(self.host, self.port) as client:
                # Prepare transcribe request
                transcribe_request = Transcribe(rate=sample_rate)
                
                # Convert audio to chunks
                chunks = wav_to_chunks(
                    audio_data.tobytes(), 
                    sample_width=2,  # 16-bit
                    rate=sample_rate, 
                    channels=1  # Mono
                )
                
                # Send transcribe request and audio chunks
                await client.write(transcribe_request)
                
                for chunk in chunks:
                    await client.write(chunk)
                
                # Signal end of audio
                await client.write(AudioChunk(audio=b'', rate=sample_rate, done=True))
                
                # Collect transcription
                transcription = ""
                async for event in client:
                    if isinstance(event, Transcript):
                        transcription += event.text + " "
                    elif event.type == "error":
                        self.logger.error(f"Transcription error: {event}")
                        break
                
                return transcription.strip()
        
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""

    async def transcribe_segments(self, audio_data, sr=16000):
        """
        Transcribe audio
        
        :param audio_data: Input audio numpy array
        :param sr: Sample rate
        :return: List of transcriptions
        """
        # Diagnostic logging of input audio
        diagnose_audio_data(audio_data, prefix='input_audio')
        
        # Save original input at original sample rate
        original_wav_path = save_wav_to_temp(
            audio_data, 
            sr, 
            prefix='raw_audio', 
            processing_stage='original',
            segment_number=0
        )
        
        # Prepare audio for transcription
        transcribe_data = audio_data
        transcribe_sr = sr
        
        # Resample if necessary
        if sr != 16000:
            self.logger.warning(f"Resampling audio from {sr} Hz to 16000 Hz")
            
            # Use resample_audio function
            resampled_results = resample_audio(audio_data, sr)
            
            # Choose one of the resampling methods (e.g., scipy_poly)
            transcribe_data = resampled_results['scipy_poly']
            transcribe_sr = 16000
            
            # Save resampled audio
            save_wav_to_temp(
                transcribe_data, 
                transcribe_sr, 
                prefix='preprocessed_audio', 
                processing_stage='resampled',
                segment_number=0
            )
        
        # Transcribe the entire audio
        transcription = await self.transcribe_audio(transcribe_data, transcribe_sr)
        
        return [transcription] if transcription else []

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
        except Exception as save_error:
            logging.error(f"Failed to save WAV: {save_error}")
            return jsonify({"error": "Could not save audio file"}), 500
        
        # Initialize transcriber
        transcriber = WhisperTranscriber()
        
        # Run transcription
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logging.info("Starting transcription")
            transcription = loop.run_until_complete(transcriber.transcribe_audio(audio_array, framerate))
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