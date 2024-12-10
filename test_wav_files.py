"""
Utility script to test WAV files with Wyoming protocol.

This script allows you to manually test WAV files recorded from the browser
against the Whisper Docker service.
"""

import asyncio
import wave
import logging
import os
import argparse

from wyoming.audio import wav_to_chunks
from wyoming.client import AsyncTcpClient
from wyoming.event import Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def send_wav_file(audio_path: str, host: str = 'localhost', port: int = 10300):
    """
    Send a WAV file to a Wyoming protocol service.
    
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
            n_frames = wav_file.getnframes()
            
            logger.info(f"WAV File Details: {os.path.basename(audio_path)}")
            logger.info(f"Rate: {rate}, Width: {width}, Channels: {channels}, Frames: {n_frames}")
            
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
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main entry point for testing WAV files."""
    parser = argparse.ArgumentParser(description="Test WAV files with Wyoming protocol")
    parser.add_argument("wav_file", help="Path to the WAV file to test")
    parser.add_argument("--host", default="localhost", help="Wyoming service host")
    parser.add_argument("--port", type=int, default=10300, help="Wyoming service port")
    
    args = parser.parse_args()
    
    asyncio.run(send_wav_file(args.wav_file, args.host, args.port))

if __name__ == '__main__':
    main()
