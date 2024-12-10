// Audio Processor for Voice Activity Detection
class WhisperAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        
        // VAD Configuration
        this.VAD_CONFIG = {
            SPEECH_THRESHOLD: 0.01,    // Minimum RMS to consider as speech
            SILENCE_THRESHOLD: 0.0005, // Ultra-low threshold for silence
            MAX_RECORDING_DURATION: 10000, // Maximum recording duration (ms)
            MAX_AUDIO_DURATION: 10,    // Max audio duration in seconds
            TAIL_DURATION: 500         // Add 500ms tail after speech ends
        };

        // State tracking
        this.isSpeechDetected = false;
        this.speechStartTime = 0;
        this.audioBuffer = [];
        this.peakRMS = 0;
        this.lastSpeechEndTime = 0;
    }

    calculateRMS(inputData) {
        return Math.sqrt(
            inputData.reduce((sum, sample) => sum + sample * sample, 0) / inputData.length
        );
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        // Ensure we have input data
        if (!input.length || !input[0].length) {
            return true;
        }

        const inputData = input[0];
        const currentTime = Date.now();
        
        // Calculate RMS
        const rms = this.calculateRMS(inputData);
        
        // Update peak RMS
        this.peakRMS = Math.max(this.peakRMS, rms);

        // Speech Detection Logic
        if (rms > this.VAD_CONFIG.SPEECH_THRESHOLD) {
            // Speech started
            if (!this.isSpeechDetected) {
                this.isSpeechDetected = true;
                this.speechStartTime = currentTime;
                
                // Notify main thread about speech start
                this.port.postMessage({
                    type: 'speech_start',
                    rms: rms,
                    timestamp: currentTime
                });
            }
            
            // Accumulate audio data
            this.audioBuffer.push(...inputData);
            
            // Prevent unbounded growth
            const maxBufferSize = 128 * this.VAD_CONFIG.MAX_AUDIO_DURATION;
            if (this.audioBuffer.length > maxBufferSize) {
                this.audioBuffer = this.audioBuffer.slice(-maxBufferSize);
            }
        } 
        else if (rms <= this.VAD_CONFIG.SILENCE_THRESHOLD && this.isSpeechDetected) {
            // Check if we've been silent long enough to end speech
            const timeSinceSpeechStart = currentTime - this.speechStartTime;
            const timeSinceLastSpeech = currentTime - this.lastSpeechEndTime;
            
            // Only end speech if we've been silent for a while and not too recently
            if (timeSinceSpeechStart > 500 && timeSinceLastSpeech > 1000) {
                const speechDuration = currentTime - this.speechStartTime;
                
                // Send audio if buffer is not empty and within duration
                if (this.audioBuffer.length > 0 && speechDuration < this.VAD_CONFIG.MAX_RECORDING_DURATION) {
                    // Convert audio buffer to Float32Array
                    const audioData = new Float32Array(this.audioBuffer);
                    
                    // Notify main thread about speech end and audio data
                    this.port.postMessage({
                        type: 'speech_end',
                        audioData: audioData,
                        speechDuration: speechDuration,
                        rms: rms,
                        audioBufferLength: audioData.length,
                        timestamp: currentTime
                    });
                }
                
                // Reset state
                this.lastSpeechEndTime = currentTime;
                this.audioBuffer = [];
                this.isSpeechDetected = false;
                this.speechStartTime = 0;
            }
        }

        return true;
    }
}

// Register the processor
registerProcessor('whisper-audio-processor', WhisperAudioProcessor);
