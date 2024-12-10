import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

def load_wav(filepath):
    """
    Load WAV file and return audio data with sample rate
    """ 
    audio_data, sample_rate = sf.read(filepath)
    print(f"Loaded audio:")
    print(f"  Filepath: {filepath}")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Shape: {audio_data.shape}")
    print(f"  Dtype: {audio_data.dtype}")
    print(f"  Min: {audio_data.min()}")
    print(f"  Max: {audio_data.max()}")
    return audio_data, sample_rate

def resample_audio(audio_data, orig_sr, target_sr=16000):
    """
    Resample audio using different methods for comparison
    
    :param audio_data: Input audio numpy array
    :param orig_sr: Original sample rate
    :param target_sr: Target sample rate
    :return: Dictionary of resampled audio arrays
    """
    # Ensure input is a numpy array
    audio_data = np.asarray(audio_data)
    
    # Normalize input audio to [-1, 1] range
    print("Pre-Normalization Input Audio:")
    print(f"  Min: {audio_data.min()}")
    print(f"  Max: {audio_data.max()}")
    print(f"  Mean: {audio_data.mean()}")
    print(f"  Std Dev: {audio_data.std()}")
    
    # Normalize if not already in correct range
    if audio_data.min() < -1 or audio_data.max() > 1:
        print("Normalizing input audio...")
        audio_data = audio_data / max(abs(audio_data.min()), abs(audio_data.max()))
    
    # Compute ratio
    ratio = target_sr / orig_sr

    # Method 1: scipy.signal.resample
    scipy_resampled = signal.resample(
        audio_data, 
        num=int(len(audio_data) * ratio)
    )
    
    # Method 2: scipy.signal.resample_poly (potentially higher quality)
    scipy_poly_resampled = signal.resample_poly(
        audio_data, 
        up=target_sr, 
        down=orig_sr
    )
    
    # Log resampled audio details
    print("\nResampled Audio Details:")
    
    print("scipy_basic:")
    print(f"  Shape: {scipy_resampled.shape}")
    print(f"  Min: {scipy_resampled.min()}")
    print(f"  Max: {scipy_resampled.max()}")
    print(f"  Mean: {scipy_resampled.mean()}")
    print(f"  Std Dev: {scipy_resampled.std()}")
    
    print("\nscipy_poly:")
    print(f"  Shape: {scipy_poly_resampled.shape}")
    print(f"  Min: {scipy_poly_resampled.min()}")
    print(f"  Max: {scipy_poly_resampled.max()}")
    print(f"  Mean: {scipy_poly_resampled.mean()}")
    print(f"  Std Dev: {scipy_poly_resampled.std()}")

    return {
        'scipy_basic': scipy_resampled,
        'scipy_poly': scipy_poly_resampled
    }

def save_wav(audio_data, sample_rate, filepath):
    """
    Save audio data to WAV file
    """
    try:
        sf.write(filepath, audio_data, sample_rate)
        print(f"Saved WAV file: {filepath}")
        print(f"  Shape: {audio_data.shape}")
        print(f"  Min: {audio_data.min()}")
        print(f"  Max: {audio_data.max()}")
        print(f"  Sample Rate: {sample_rate}")
        print(f"  Data Type: {audio_data.dtype}")
    except Exception as e:
        print(f"Error saving WAV file {filepath}: {e}")

def plot_comparison(original, resampled_dict, orig_sr, target_sr):
    """
    Plot original and resampled audio for comparison
    """
    plt.figure(figsize=(15, 10))

    # Original audio
    plt.subplot(3, 1, 1)
    plt.title(f'Original Audio ({orig_sr} Hz)')
    plt.plot(original)
    plt.ylabel('Amplitude')

    # Resampled audios
    for i, (method, resampled) in enumerate(resampled_dict.items(), 2):
        plt.subplot(3, 1, i)
        plt.title(f'Resampled Audio - {method} ({target_sr} Hz)')
        plt.plot(resampled)
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('resampling_comparison.png')
    plt.close()

def main():
    # Source file path
    source_file = r'c:/_dev/my_whisper/_temp/raw_audio_44100hz_20241210_130803_465409_original.wav'
    
    # Ensure output directory exists
    output_dir = r'c:/_dev/my_whisper/_temp/resampled'
    os.makedirs(output_dir, exist_ok=True)

    # Load original audio
    original_audio, orig_sr = load_wav(source_file)

    # Resample
    target_sr = 16000
    resampled_audio = resample_audio(original_audio, orig_sr, target_sr)

    # Save resampled files
    output_path = os.path.join(output_dir, f'resampled_poly_{target_sr}hz.wav')
    save_wav(resampled_audio['scipy_poly'], target_sr, output_path)

    # Plot comparison
    plot_comparison(original_audio, resampled_audio, orig_sr, target_sr)

if __name__ == '__main__':
    main()
