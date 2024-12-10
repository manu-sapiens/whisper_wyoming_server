import soundfile as sf
import numpy as np

# Load the audio file
audio_data, sample_rate = sf.read('test.wav')

print(f"Audio Data Shape: {audio_data.shape}")
print(f"Sample Rate: {sample_rate} Hz")
print(f"Duration: {len(audio_data) / sample_rate} seconds")
print(f"Min Amplitude: {audio_data.min()}")
print(f"Max Amplitude: {audio_data.max()}")
print(f"Mean Amplitude: {audio_data.mean()}")
print(f"RMS Amplitude: {np.sqrt(np.mean(audio_data**2))}")

# Analyze amplitude over time
chunk_size = sample_rate // 10  # 100ms chunks
chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
chunk_rms = [np.sqrt(np.mean(chunk**2)) for chunk in chunks]

print("\nChunk RMS Values:")
for i, rms in enumerate(chunk_rms):
    print(f"Chunk {i}: {rms}")
