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
    :return: Path to saved WAV file
    """
    # Ensure _temp directory exists
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '_temp'))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Use precise timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Construct more descriptive filename
    # Include sample rate and processing details
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
        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Normalize if not already in correct range
        if audio_array.min() < -1 or audio_array.max() > 1:
            logging.warning("Normalizing audio array")
            audio_data = audio_array / max(abs(audio_array.min()), abs(audio_array.max()))
        else:
            audio_data = audio_array
        
        # Write file with soundfile
        sf.write(
            filepath, 
            audio_data, 
            sample_rate, 
            subtype='PCM_16'  # Explicitly set 16-bit PCM
        )
        
        # Verify file
        with sf.SoundFile(filepath, 'r') as sound_file:
            logging.info(f"Saved WAV file details:")
            logging.info(f"Filename: {filename}")
            logging.info(f"Channels: {sound_file.channels}")
            logging.info(f"File size: {os.path.getsize(filepath)} bytes")
            
        return filepath
    
    except Exception as e:
        logging.error(f"Failed to save WAV file: {e}")
        raise

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
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sample_rate, 
                target_sr=16000
            )
            sample_rate = 16000  # Update to resampled rate
        
        # Convert audio to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        try:
            # Establish connection to Wyoming service
            async with AsyncTcpClient(self.host, self.port) as client:
                # Send transcribe request
                await client.send(Transcribe(rate=sample_rate))
                
                # Convert audio to chunks
                chunks = wav_to_chunks(
                    audio_data.tobytes(), 
                    sample_width=2,  # 16-bit
                    rate=sample_rate, 
                    channels=1  # Mono
                )
                
                # Send audio chunks
                for chunk in chunks:
                    await client.send(chunk)
                
                # Signal end of audio
                await client.send(AudioChunk(audio=b'', rate=sample_rate, done=True))
                
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
        Transcribe voice segments
        
        :param audio_data: Input audio numpy array
        :param sr: Sample rate
        :return: List of transcriptions
        """
        # Diagnostic logging of input audio
        diagnose_audio_data(audio_data, prefix='input_audio')
        
        # Detailed resampling diagnostics
        original_sr = sr
        original_length = len(audio_data)
        
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Save original input
        save_wav_to_temp(
            audio_data, 
            original_sr, 
            prefix='preprocessed_audio', 
            processing_stage='original',
            segment_number=0  # Mark as initial input
        )
        
        # Resample if necessary
        if sr != 16000:
            self.logger.warning(f"Resampling audio from {sr} Hz to 16000 Hz")
            
            # Pre-resampling diagnostics
            pre_resample_stats = {
                'min': audio_data.min(),
                'max': audio_data.max(),
                'mean': audio_data.mean(),
                'std': audio_data.std()
            }
            
            # Perform resampling
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
            
            # Post-resampling diagnostics
            post_resample_stats = {
                'min': audio_data.min(),
                'max': audio_data.max(),
                'mean': audio_data.mean(),
                'std': audio_data.std()
            }
            
            # Log detailed resampling information
            self.logger.info("Resampling Diagnostic Report:")
            self.logger.info(f"Original Sample Rate: {original_sr} Hz")
            self.logger.info(f"Target Sample Rate: {sr} Hz")
            self.logger.info(f"Original Audio Length: {original_length} samples")
            self.logger.info(f"Resampled Audio Length: {len(audio_data)} samples")
            
            self.logger.info("Pre-Resampling Statistics:")
            for key, value in pre_resample_stats.items():
                self.logger.info(f"  {key}: {value}")
            
            self.logger.info("Post-Resampling Statistics:")
            for key, value in post_resample_stats.items():
                self.logger.info(f"  {key}: {value}")
            
            # Save resampled audio
            save_wav_to_temp(
                audio_data, 
                sr, 
                prefix='preprocessed_audio', 
                processing_stage='resampled',
                segment_number=0  # Mark as initial resampled input
            )
        
        # Detect voice activity
        voice_segments = detect_voice_activity(audio_data, sr)
        
        transcriptions = []
        saved_segment_paths = []
        
        for i, (start, end) in enumerate(voice_segments):
            # Extract segment
            segment = audio_data[start:end]
            
            # Diagnostic print
            print(f"Segment {i} Details:")
            print(f"Start: {start}, End: {end}")
            print(f"Segment Length: {len(segment)}")
            print(f"Segment Mean: {segment.mean()}")
            print(f"Segment Std Dev: {segment.std()}")
            
            # Save segment for diagnostic purposes
            segment_wav_path = save_wav_to_temp(
                segment, 
                sr, 
                prefix='segment', 
                segment_number=i,  # Use explicit segment number 
                processing_stage='extracted'
            )
            saved_segment_paths.append(segment_wav_path)
            print(f"Saved Segment WAV: {segment_wav_path}")
            
            # Async transcription
            try:
                result_container = []
                transcription_thread = threading.Thread(
                    target=run_async_transcription, 
                    args=(self, segment, sr, result_container)
                )
                transcription_thread.start()
                transcription_thread.join(timeout=10)  # 10-second timeout
                
                if result_container:
                    transcriptions.append(result_container[0])
                else:
                    self.logger.warning(f"No transcription for segment {i}")
            
            except Exception as e:
                self.logger.error(f"Transcription error for segment {i}: {e}")
    
        return transcriptions

def detect_voice_activity(audio_data, sr=16000, frame_duration=30, aggressiveness=3):
    """
    Perform Voice Activity Detection using WebRTC VAD
    
    :param audio_data: Input audio numpy array (float32)
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
    logger.info(f"Audio Min: {audio_data.min()}")
    logger.info(f"Audio Max: {audio_data.max()}")
    logger.info(f"Audio Mean: {audio_data.mean()}")
    logger.info(f"Audio Std Dev: {audio_data.std()}")
    
    # Validate sample rate
    valid_sample_rates = [8000, 16000, 32000, 48000]
    if sr not in valid_sample_rates:
        logger.warning(f"Sample rate {sr} not in {valid_sample_rates}. Resampling to 16000 Hz")
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Ensure audio is float32 and normalized
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Normalize audio to prevent VAD issues
    audio_data = audio_data / np.max(np.abs(audio_data))
    
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