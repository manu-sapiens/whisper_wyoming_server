import os
import wave
import numpy as np
import matplotlib.pyplot as plt

def compare_wav_files(good_file, bad_file):
    """
    Compare two WAV files in detail
    """
    # Basic file information
    print("File Comparison:")
    print(f"Good file: {good_file}")
    print(f"Bad file: {bad_file}")
    
    # Open WAV files
    with wave.open(good_file, 'rb') as good_wav, wave.open(bad_file, 'rb') as bad_wav:
        # WAV file parameters
        print("\nWAV File Parameters:")
        good_params = {
            'Channels': good_wav.getnchannels(),
            'Sample Width': good_wav.getsampwidth(),
            'Framerate': good_wav.getframerate(),
            'Frames': good_wav.getnframes()
        }
        bad_params = {
            'Channels': bad_wav.getnchannels(),
            'Sample Width': bad_wav.getsampwidth(),
            'Framerate': bad_wav.getframerate(),
            'Frames': bad_wav.getnframes()
        }
        
        # Print parameters
        for key in good_params:
            print(f"{key}: Good = {good_params[key]}, Bad = {bad_params[key]}")
        
        # Read audio data
        good_data = good_wav.readframes(good_params['Frames'])
        bad_data = bad_wav.readframes(bad_params['Frames'])
    
    # Convert to numpy arrays
    good_array = np.frombuffer(good_data, dtype=np.int16)
    bad_array = np.frombuffer(bad_data, dtype=np.int16)
    
    # Detailed statistical analysis
    print("\nStatistical Analysis:")
    print("Good File:")
    print(f"  Min: {good_array.min()}")
    print(f"  Max: {good_array.max()}")
    print(f"  Mean: {good_array.mean()}")
    print(f"  Std Dev: {good_array.std()}")
    
    print("\nBad File:")
    print(f"  Min: {bad_array.min()}")
    print(f"  Max: {bad_array.max()}")
    print(f"  Mean: {bad_array.mean()}")
    print(f"  Std Dev: {bad_array.std()}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Waveform plot
    plt.subplot(2, 2, 1)
    plt.title('Good File Waveform')
    plt.plot(good_array)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 2, 2)
    plt.title('Bad File Waveform')
    plt.plot(bad_array)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Histogram
    plt.subplot(2, 2, 3)
    plt.title('Good File Amplitude Distribution')
    plt.hist(good_array, bins=50)
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 4)
    plt.title('Bad File Amplitude Distribution')
    plt.hist(bad_array, bins=50)
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('wav_comparison.png')
    plt.close()

def main():
    # Directory containing the files
    temp_dir = 'c:/_dev/my_whisper/_temp'
    
    # Find WAV files
    wav_files = [
        os.path.join(temp_dir, f) 
        for f in os.listdir(temp_dir) 
        if f.endswith('.wav')
    ]
    
    # Sort files by modification time, most recent first
    wav_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # If we have at least 2 WAV files, compare the most recent two
    if len(wav_files) >= 2:
        print(f"Comparing most recent WAV files:")
        print(f"File 1: {wav_files[0]}")
        print(f"File 2: {wav_files[1]}")
        compare_wav_files(wav_files[0], wav_files[1])
    else:
        print("Not enough WAV files to compare.")

if __name__ == '__main__':
    main()
