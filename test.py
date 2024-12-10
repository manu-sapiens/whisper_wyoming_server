import asyncio
import json
import logging
import soundfile as sf
import wave

from wyoming.audio import AudioChunk, AudioChunkConverter, wav_to_chunks
from wyoming_audio import convert_audio_to_wyoming_chunk, create_test_wav_file

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def describe_service(host='127.0.0.1', port=10300):
    """
    Send a 'describe' message to get service information
    """
    print("\n===== Wyoming Protocol Service Description =====")
    
    try:
        # Create WebSocket connection
        reader, writer = await asyncio.open_connection(host, port)
        
        # Prepare describe message
        describe_msg = json.dumps({"type": "describe"}) + "\n"
        writer.write(describe_msg.encode())
        await writer.drain()
        
        # Read response
        response = await reader.readline()
        response_json = json.loads(response.decode().strip())
        
        if response_json.get('type') == 'info':
            print("Service Information:")
            if 'asr' in response_json:
                print("ASR (Speech Recognition) Services:")
                for model in response_json['asr'].get('models', []):
                    print(f"- Name: {model.get('name')}")
                    print(f"  Languages: {model.get('languages')}")
                    print(f"  Attribution: {model.get('attribution')}")
        
        writer.close()
        await writer.wait_closed()
    
    except Exception as e:
        logger.error(f"Error describing service: {e}")
    
    print("=============================================\n")

async def test_wyoming_transcription(audio_path):
    """
    Test transcription using Wyoming protocol
    """
    # Read audio file
    try:
        audio_data, sample_rate = sf.read(audio_path)
    except Exception as e:
        logger.error(f"Failed to read audio file: {e}")
        print(f"Error reading audio file: {audio_path}")
        return

    logger.info(f"Loaded audio - Sample rate: {sample_rate}")
    
    # Convert audio to Wyoming chunk
    wyoming_chunk = convert_audio_to_wyoming_chunk(audio_data)
    
    print("\n===== Wyoming Protocol Transcription =====")
    
    try:
        # Create WebSocket connection
        reader, writer = await asyncio.open_connection('127.0.0.1', 10300)
        
        # Prepare transcription request
        transcribe_event = json.dumps({
            "type": "transcribe",
            "data": {
                "audio_format": {
                    "rate": wyoming_chunk.rate,
                    "width": wyoming_chunk.width,
                    "channels": wyoming_chunk.channels
                }
            }
        }) + "\n"
        
        # Send transcription request
        writer.write(transcribe_event.encode())
        
        # Send audio chunk
        writer.write(wyoming_chunk.audio)
        await writer.drain()
        
        # Wait for transcription result
        response = await reader.readline()
        response_json = json.loads(response.decode().strip())
        
        if response_json.get('type') == 'transcript':
            print("Transcription Result:")
            print(f"Text: {response_json.get('text', 'No text')}")
        
        writer.close()
        await writer.wait_closed()
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
    
    print("=============================================\n")

def test_chunk_converter() -> None:
    """Test audio chunk converter."""
    converter = AudioChunkConverter(rate=16000, width=2, channels=1)
    input_chunk = AudioChunk(
        rate=48000,
        width=4,
        channels=2,
        audio=bytes(1 * 48000 * 4 * 2),  # 1 sec
    )

    output_chunk = converter.convert(input_chunk)
    assert output_chunk.rate == 16000
    assert output_chunk.width == 2
    assert output_chunk.channels == 1
    assert len(output_chunk.audio) == 1 * 16000 * 2 * 1  # 1 sec
    print("✓ Chunk Converter Test Passed")

def test_wav_to_chunks() -> None:
    """Test WAV file to audio chunks."""
    # Create a test WAV file
    test_wav_path = 'test_audio.wav'
    create_test_wav_file(test_wav_path)

    with wave.open(test_wav_path, 'rb') as wav_read:
        chunks = list(wav_to_chunks(wav_read, samples_per_chunk=1000))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, AudioChunk)
            assert chunk.rate == 16000
            assert chunk.width == 2
            assert chunk.channels == 1
            assert len(chunk.audio) == 1000 * 2  # 1000 samples
        print("✓ WAV to Chunks Test Passed")

def main():
    # Run audio utility tests
    test_chunk_converter()
    test_wav_to_chunks()
    
    # Test Wyoming transcription
    audio_path = r"c:/_dev/my_whisper/recorded_audio.wav"
    asyncio.run(test_wyoming_transcription(audio_path))

if __name__ == '__main__':
    main()
