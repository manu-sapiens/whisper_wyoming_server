import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def load_audio_with_details(filepath):
    """
    Load audio file and provide comprehensive details
    """
    try:
        # Load audio using librosa (handles various formats)
        audio, sr = librosa.load(filepath, sr=None, mono=True)
        
        logging.info(f"Audio File: {filepath}")
        logging.info(f"Sample Rate: {sr} Hz")
        logging.info(f"Duration: {librosa.get_duration(y=audio, sr=sr):.2f} seconds")
        logging.info(f"Total Samples: {len(audio)}")
        logging.info(f"Audio Dtype: {audio.dtype}")
        logging.info(f"Min Amplitude: {audio.min()}")
        logging.info(f"Max Amplitude: {audio.max()}")
        logging.info(f"Mean Amplitude: {audio.mean()}")
        logging.info(f"Standard Deviation: {audio.std()}")
        
        return audio, sr
    
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        return None, None

def compare_audio_characteristics(files):
    """
    Compare characteristics of multiple audio files
    """
    audio_data = []
    sample_rates = []
    
    # Load audio files
    for filepath in files:
        audio, sr = load_audio_with_details(filepath)
        if audio is not None:
            audio_data.append(audio)
            sample_rates.append(sr)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Waveform Comparison
    plt.subplot(2, 2, 1)
    plt.title('Waveform Comparison')
    for i, audio in enumerate(audio_data):
        plt.plot(audio, label=f'File {i+1}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Spectrogram Comparison
    plt.subplot(2, 2, 2)
    plt.title('Spectrogram Comparison')
    for i, (audio, sr) in enumerate(zip(audio_data, sample_rates)):
        D = librosa.stft(audio)
        plt.specgram(librosa.amplitude_to_db(np.abs(D), ref=np.max), 
                     Fs=sr, cmap='viridis')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    # Histogram of Amplitudes
    plt.subplot(2, 2, 3)
    plt.title('Amplitude Distribution')
    for i, audio in enumerate(audio_data):
        plt.hist(audio, bins=50, alpha=0.5, label=f'File {i+1}')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Resampling Comparison
    plt.subplot(2, 2, 4)
    plt.title('Resampled Audio Comparison')
    target_sr = 16000  # Whisper's preferred sample rate
    for i, (audio, sr) in enumerate(zip(audio_data, sample_rates)):
        if sr != target_sr:
            resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            plt.plot(resampled_audio, label=f'Resampled File {i+1}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('audio_diagnostics.png')
    plt.close()

def main():
    # Directory with audio files
    temp_dir = 'c:/_dev/my_whisper/_temp'
    
    # Find WAV files in the temp directory
    wav_files = [
        os.path.join(temp_dir, f) 
        for f in os.listdir(temp_dir) 
        if f.endswith('.wav')
    ]
    
    if len(wav_files) < 2:
        logging.warning("Not enough WAV files to compare. Need at least 2.")
        return
    
    # Sort files to ensure consistent order
    wav_files.sort()
    
    # Compare first two files
    compare_audio_characteristics(wav_files[:2])

if __name__ == '__main__':
    main()
