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

from flask import Flask, render_template, request, jsonify
from wyoming.client import AsyncClient, AsyncTcpClient
from wyoming.event import Event
from wyoming.audio import AudioChunk, wav_to_chunks
from wyoming.asr import Transcribe, Transcript

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)

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
        temp_wav_path = None
        try:
            # Create a temporary WAV file from NumPy array
            import tempfile
            import os

            # Generate a predictable temporary filename in the temp directory
            temp_wav_path = os.path.join(
                tempfile.gettempdir(), 
                f"whisper_recording_{int(time.time())}.wav"
            )

            # Write NumPy array to WAV file
            wav_file = wave.open(temp_wav_path, 'wb')
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Convert NumPy array to 16-bit PCM bytes
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            wav_file.writeframes(audio_bytes)
            wav_file.close()

            # Open the WAV file for reading
            with wave.open(temp_wav_path, 'rb') as wav_file:
                # Inspect WAV file details
                rate = wav_file.getframerate()
                width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                
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
                    
                    # Send audio chunks
                    chunk_count = 0
                    for chunk in wav_to_chunks(wav_file, samples_per_chunk=1000):
                        chunk_count += 1
                        self.logger.info(f"Chunk {chunk_count}: rate={chunk.rate}, width={chunk.width}, channels={chunk.channels}, audio_len={len(chunk.audio)}")
                        
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
                    
                    # Wait for and print any responses
                    transcript = ""
                    while True:
                        event = await client.read_event()
                        if event is None:
                            break
                        self.logger.info(f"Received event: {event}")
                        
                        if event.type == "transcript":
                            transcript = event.data.get("text", "")
                            break
                    
                    return transcript

        except Exception as e:
            self.logger.error(f"Error sending audio file: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""
        
        finally:
            # Ensure temporary file is always cleaned up
            try:
                if temp_wav_path and os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
            except Exception as cleanup_error:
                self.logger.error(f"File cleanup error: {cleanup_error}")

    async def transcribe_segments(self, audio_data, sr=16000):
        """
        Transcribe voice segments
        
        :param audio_data: Input audio numpy array
        :param sr: Sample rate
        :return: List of transcriptions
        """
        voice_segments = detect_voice_activity(audio_data, sr)
        
        transcriptions = []
        for start, end in voice_segments:
            segment = audio_data[start:end]
            
            # Transcribe segment
            transcription = await self.transcribe_audio(segment, sr)
            if transcription.strip():
                transcriptions.append(transcription)
        
        return transcriptions

def detect_voice_activity(audio_data, sr=16000, frame_duration=0.03, threshold=0.02):
    """
    Perform Voice Activity Detection using librosa energy-based method
    
    :param audio_data: Input audio numpy array
    :param sr: Sample rate
    :param frame_duration: Duration of each frame in seconds
    :param threshold: Energy threshold for voice detection
    :return: List of voice segments (start, end indices)
    """
    # Calculate frame length
    frame_length = int(sr * frame_duration)
    
    # Compute RMS energy for each frame
    rms_energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=frame_length)[0]
    
    # Detect voice segments
    voice_segments = []
    in_voice_segment = False
    start_index = 0
    
    for i, energy in enumerate(rms_energy):
        if energy > threshold and not in_voice_segment:
            # Start of a voice segment
            start_index = i * frame_length
            in_voice_segment = True
        elif energy <= threshold and in_voice_segment:
            # End of a voice segment
            end_index = i * frame_length
            voice_segments.append((start_index, end_index))
            in_voice_segment = False
    
    # Handle case where last segment is a voice segment
    if in_voice_segment:
        voice_segments.append((start_index, len(audio_data)))
    
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
    
    # Save file temporarily with a unique name
    import tempfile
    import os
    
    try:
        # Create a temporary file with .wav extension
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as temp_file:
            file.save(temp_file)
            temp_file_path = temp_file.name
        
        # Debug: print file details
        file_size = os.path.getsize(temp_file_path)
        logging.info(f"Saved temporary file: {temp_file_path}, Size: {file_size} bytes")
        
        # Read file contents for debugging
        with open(temp_file_path, 'rb') as f:
            file_bytes = f.read()
            logging.info(f"First 50 bytes of file: {file_bytes[:50].hex()}")
        
        # Verify file is a valid WAV
        try:
            with wave.open(temp_file_path, 'rb') as wav_file:
                # Log file details
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                logging.info(f"WAV File Details: "
                             f"Channels: {n_channels}, "
                             f"Sample Width: {sample_width}, "
                             f"Framerate: {framerate}, "
                             f"Frames: {n_frames}")
                
                # Read audio data
                audio_data = wav_file.readframes(n_frames)
        except Exception as wav_error:
            # If wave module fails, try reading raw file contents
            logging.error(f"WAV file read error: {wav_error}")
            
            # Detailed file content logging
            with open(temp_file_path, 'rb') as f:
                content = f.read()
                logging.error(f"File content (first 200 bytes): {content[:200].hex()}")
            
            return jsonify({"error": f"Invalid WAV file: {str(wav_error)}"}), 400
        
        # Convert to numpy array
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as convert_error:
            logging.error(f"Audio conversion error: {convert_error}")
            return jsonify({"error": f"Could not convert audio: {str(convert_error)}"}), 400
        
        # Initialize transcriber
        transcriber = WhisperTranscriber()
        
        # Run transcription
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            transcription = loop.run_until_complete(transcriber.transcribe_audio(audio_array, framerate))
            loop.close()
        except Exception as transcribe_error:
            logging.error(f"Transcription error: {transcribe_error}")
            return jsonify({"error": f"Transcription failed: {str(transcribe_error)}"}), 500
        
        return jsonify({
            "transcription": transcription,
            "language": "auto",
            "sample_rate": framerate
        })
    
    except Exception as e:
        logging.error(f"Unexpected error in transcription: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Ensure temporary file is always cleaned up
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception as cleanup_error:
            logging.error(f"File cleanup error: {cleanup_error}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)