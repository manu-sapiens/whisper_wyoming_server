import asyncio
import logging
import numpy as np
import torch
import whisper

from wyoming.server import AsyncTcpServer
from wyoming.audio import AudioChunk, AudioChunkConverter
from wyoming.event import Event
from wyoming.asr import Transcribe, Transcript

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WhisperService:
    def __init__(self, model_name='base'):
        """
        Initialize Whisper service with specified model.
        
        Args:
            model_name (str, optional): Name of Whisper model. Defaults to 'base'.
        """
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        self.audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        
    async def transcribe_audio(self, audio_chunks):
        """
        Transcribe audio chunks using Whisper.
        
        Args:
            audio_chunks (list): List of audio chunks to transcribe
        
        Returns:
            dict: Transcription result
        """
        # Combine audio chunks
        audio_data = b''.join(chunk.audio for chunk in audio_chunks)
        
        # Convert to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        try:
            # Transcribe
            result = self.model.transcribe(audio_np)
            
            logger.info(f"Transcription result: {result['text']}")
            return {
                'text': result.get('text', '').strip(),
                'language': result.get('language', 'unknown')
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return {"text": "", "language": "unknown"}

class WhisperHandler:
    def __init__(self, whisper_service):
        """
        Initialize Wyoming protocol handler for Whisper service.
        
        Args:
            whisper_service (WhisperService): Whisper transcription service
        """
        self.whisper_service = whisper_service
        self.audio_chunks = []
        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
    
    async def handle_event(self, event):
        """
        Handle incoming Wyoming protocol events.
        
        Args:
            event (Event): Incoming event to process
        
        Returns:
            bool: Whether to continue processing events
        """
        logger.debug(f"Received event: {event}")
        
        if Transcribe.is_type(event.type):
            # Reset audio chunks when a new transcription request is received
            self.audio_chunks = []
            return True
        
        if event.type == 'audio-start':
            # Reset audio chunks on audio start
            self.audio_chunks = []
            return True
        
        if event.type == 'audio-chunk':
            # Convert and store audio chunk
            try:
                chunk = AudioChunk(
                    rate=event.data.get('rate', 16000),
                    width=event.data.get('width', 2),
                    channels=event.data.get('channels', 1),
                    audio=event.payload
                )
                
                # Convert chunk if needed
                converted_chunk = self.converter.convert(chunk)
                self.audio_chunks.append(converted_chunk)
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            return True
        
        if event.type == 'audio-stop':
            # Transcribe collected audio chunks
            try:
                result = await self.whisper_service.transcribe_audio(self.audio_chunks)
                
                # Send transcription result
                logger.info(f"Sending transcript event: {result}")
                await self.write_event(Transcript(text=result['text']).event())
            except Exception as e:
                logger.error(f"Transcription error: {e}", exc_info=True)
                await self.write_event(Transcript(text='').event())
            
            # Reset audio chunks
            self.audio_chunks = []
            return False
        
        return True

async def main():
    """
    Start Wyoming protocol Whisper service.
    """
    whisper_service = WhisperService()
    
    # Create Wyoming TCP server
    server = AsyncTcpServer.from_uri('tcp://0.0.0.0:10300')
    
    # Start server with Whisper handler
    logger.info("Starting Whisper service on tcp://0.0.0.0:10300")
    await server.start(lambda: WhisperHandler(whisper_service))

if __name__ == '__main__':
    asyncio.run(main())
