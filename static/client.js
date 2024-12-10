// Custom VAD Configuration
const VAD_CONFIG = {
    SPEECH_THRESHOLD: 0.01,    // Minimum RMS to consider as speech
    SILENCE_THRESHOLD: 0.0005, // Ultra-low threshold for silence
    MAX_RECORDING_DURATION: 10000 // Maximum recording duration (ms)
};

// Whisper Service Configuration
const WHISPER_CONFIG = {
    HOST: 'http://127.0.0.1',
    PORT: 5000,
    TIMEOUT: 5000,
    ENDPOINTS: {
        TRANSCRIBE: '/transcribe'
    },
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 2000
};

let audioContext;
let mediaStream;
let analyser;
let javascriptNode;
let isRecording = false;
let peakRMS = 0;  // Track peak RMS globally

// Custom VAD state tracking
let audioBuffer = [];
let isSpeechDetected = false;
let speechStartTime = 0;

const statusDiv = document.getElementById('status');
const debugDiv = document.getElementById('debug');
const transcriptionDiv = document.getElementById('transcriptionResult');
const rmsDisplayDiv = document.getElementById('rmsDisplay');
const audioBufferDisplayDiv = document.getElementById('audioBufferDisplay');
const whisperStatusDiv = document.getElementById('whisperStatus');

function logDebug(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const formattedMessage = `[${timestamp}] ${message}`;
    
    console.log(formattedMessage);
    
    const messageSpan = document.createElement('span');
    messageSpan.textContent = formattedMessage + '\n';
    messageSpan.classList.add(type);
    debugDiv.appendChild(messageSpan);
    debugDiv.scrollTop = debugDiv.scrollHeight;
}

// Handshake with Whisper Docker service
async function checkWhisperService() {
    for (let attempt = 1; attempt <= WHISPER_CONFIG.RETRY_ATTEMPTS; attempt++) {
        try {
            const startTime = Date.now();
            
            // Try multiple connection methods
            const connectionMethods = [
                `${WHISPER_CONFIG.HOST}:${WHISPER_CONFIG.PORT}${WHISPER_CONFIG.ENDPOINTS.TRANSCRIBE}`,
                `http://localhost:${WHISPER_CONFIG.PORT}${WHISPER_CONFIG.ENDPOINTS.TRANSCRIBE}`,
                `http://127.0.0.1:${WHISPER_CONFIG.PORT}${WHISPER_CONFIG.ENDPOINTS.TRANSCRIBE}`
            ];

            let response;
            for (const url of connectionMethods) {
                try {
                    response = await Promise.race([
                        fetch(url, {
                            method: 'GET',
                            mode: 'no-cors',  // Attempt to bypass CORS
                            headers: {
                                'Accept': 'application/json'
                            }
                        }),
                        new Promise((_, reject) => 
                            setTimeout(() => reject(new Error('Timeout')), WHISPER_CONFIG.TIMEOUT)
                        )
                    ]);

                    if (response.ok || response.type === 'opaque') {
                        break;
                    }
                } catch (connectionError) {
                    logDebug(`Connection attempt failed for ${url}: ${connectionError.message}`, 'warning');
                    continue;
                }
            }

            const endTime = Date.now();
            const latency = endTime - startTime;

            if (response && (response.ok || response.type === 'opaque')) {
                whisperStatusDiv.textContent = `Whisper Service: Connected (Attempt: ${attempt}, Latency: ${latency}ms)`;
                whisperStatusDiv.style.color = 'green';
                logDebug(`Whisper Service Handshake Successful on attempt ${attempt}`, 'info');
                return true;
            } else {
                throw new Error('No successful connection');
            }
        } catch (error) {
            logDebug(`Whisper Service Connection Attempt ${attempt} Failed: ${error.message}`, 'warning');
            
            whisperStatusDiv.textContent = `Whisper Service: Connecting... (Attempt ${attempt}/${WHISPER_CONFIG.RETRY_ATTEMPTS})`;
            whisperStatusDiv.style.color = 'orange';

            // Wait before retrying
            if (attempt < WHISPER_CONFIG.RETRY_ATTEMPTS) {
                await new Promise(resolve => setTimeout(resolve, WHISPER_CONFIG.RETRY_DELAY));
            }
        }
    }

    // Final failure state
    whisperStatusDiv.textContent = 'Whisper Service: Unreachable';
    whisperStatusDiv.style.color = 'red';
    logDebug('All Whisper Service Connection Attempts Failed', 'error');
    return false;
}

async function sendAudioToServer(audioData, sampleRate) {
    logDebug(`Sending audio segment: ${audioData.length} samples`);
    logDebug(`Audio details - Sample Rate: ${sampleRate}, Target Rate: 16000, Blob Size: ${audioData.byteLength} bytes`);

    try {
        // Create WAV file from audio data
        const wavBuffer = float32ArrayToWav(audioData, sampleRate);
        
        // Debug: log WAV file details
        logDebug(`WAV Buffer created: ${wavBuffer.byteLength} bytes`);
        
        // Create Blob from WAV buffer
        const audioBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        
        // Debug: log Blob details
        logDebug(`Blob created: ${audioBlob.size} bytes, Type: ${audioBlob.type}`);
        
        // Save WAV file to disk for debugging
        const a = document.createElement('a');
        const url = URL.createObjectURL(audioBlob);
        a.href = url;
        
        // Generate filename with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        a.download = `recording_${timestamp}.wav`;
        
        // Trigger download
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');

        // Send to local server endpoint
        const response = await fetch(`${WHISPER_CONFIG.HOST}:${WHISPER_CONFIG.PORT}${WHISPER_CONFIG.ENDPOINTS.TRANSCRIBE}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Update UI with transcription
        transcriptionDiv.textContent = result.transcription || 'No transcription';
        logDebug(`Transcription received. Language: ${result.language}`);

        return result.transcription;
    } catch (error) {
        logDebug(`Transcription error: ${error.message}`, 'error');
        transcriptionDiv.textContent = `Transcription failed: ${error.message}`;
        throw error;
    }
}

// Convert Float32Array to WAV format
function float32ArrayToWav(samples, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    // Validate input
    if (!samples || samples.length === 0) {
        logDebug('No audio samples provided', 'error');
        throw new Error('No audio samples provided');
    }
    
    // Debug: log input samples
    logDebug(`Converting ${samples.length} samples to WAV. Sample Rate: ${sampleRate}`);
    
    // Convert float samples to 16-bit PCM
    const int16Samples = new Int16Array(samples.length);
    let minSample = Infinity, maxSample = -Infinity;
    
    for (let i = 0; i < samples.length; i++) {
        // Clamp and scale float samples to 16-bit range
        const sample = Math.max(-1, Math.min(1, samples[i]));
        int16Samples[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        
        // Track min/max for debugging
        minSample = Math.min(minSample, sample);
        maxSample = Math.max(maxSample, sample);
    }
    
    // Debug: log sample range
    logDebug(`Sample Range: Min=${minSample}, Max=${maxSample}`);
    
    // Calculate total file size
    const dataSize = int16Samples.length * bytesPerSample;
    const fileSize = 44 + dataSize;
    
    // Create WAV header
    const buffer = new ArrayBuffer(fileSize);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, fileSize - 8, true);  // File size minus RIFF header
    writeString(view, 8, 'WAVE');

    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);  // Chunk size
    view.setUint16(20, 1, true);   // Audio format (PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Write audio data
    const dataView = new Int16Array(buffer, 44);
    dataView.set(int16Samples);

    // Debug: verify WAV header
    logDebug(`WAV Header: 
        File Size: ${fileSize} bytes
        RIFF Chunk Size: ${view.getUint32(4, true)}
        Sample Rate: ${view.getUint32(24, true)}
        Bytes per Second: ${view.getUint32(28, true)}
        Block Align: ${view.getUint16(32, true)}
        Bits per Sample: ${view.getUint16(34, true)}
        Data Size: ${view.getUint32(40, true)}`);

    return buffer;
}

// Helper function to write strings to DataView
function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

async function startAdvancedRecording() {
    try {
        // Check Whisper service before starting recording
        const whisperReady = await checkWhisperService();
        if (!whisperReady) {
            statusDiv.textContent = 'Cannot start recording: Whisper service unavailable';
            return;
        }

        // Reset tracking variables
        peakRMS = 0;
        audioBuffer = [];
        isSpeechDetected = false;
        speechStartTime = 0;

        // Enumerate and log devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputDevices = devices.filter(device => device.kind === 'audioinput');
        
        logDebug('Available Audio Input Devices:', 'info');
        audioInputDevices.forEach((device, index) => {
            logDebug(`[${index}] ${device.label} (${device.deviceId})`, 'info');
        });

        // Check device and permission support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('getUserMedia not supported');
        }

        // Analyze audio tracks
        const tracks = mediaStream.getAudioTracks();
        logDebug(`Number of audio tracks: ${tracks.length}`, 'info');

        tracks.forEach((track, index) => {
            const settings = track.getSettings();
            logDebug(`Track [${index}] Settings:`, 'info');
            logDebug(JSON.stringify(settings, null, 2), 'info');
        });

        // Create audio context with native sample rate
        const nativeSampleRate = tracks[0].getSettings().sampleRate;
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: nativeSampleRate
        });

        // Advanced audio analysis setup
        analyser = audioContext.createAnalyser();
        analyser.minDecibels = -90;
        analyser.maxDecibels = -10;
        analyser.smoothingTimeConstant = 0.8;
        analyser.fftSize = 2048;

        // Script processor for audio processing
        javascriptNode = audioContext.createScriptProcessor(4096, 1, 1);

        // Connect audio graph
        const source = audioContext.createMediaStreamSource(mediaStream);
        source.connect(analyser);
        analyser.connect(javascriptNode);
        javascriptNode.connect(audioContext.destination);

        javascriptNode.onaudioprocess = (event) => {
            const inputBuffer = event.inputBuffer;
            const inputData = inputBuffer.getChannelData(0);

            // Compute RMS with more robust calculation
            const rms = Math.sqrt(
                inputData.reduce((sum, sample) => sum + sample * sample, 0) / inputData.length
            );

            // Update peak RMS
            peakRMS = Math.max(peakRMS, rms);

            // Update RMS display on a single line
            rmsDisplayDiv.textContent = `Current RMS: ${rms.toFixed(4)} | Peak RMS: ${peakRMS.toFixed(4)}`;

            // Custom VAD Logic
            const currentTime = Date.now();

            if (rms > VAD_CONFIG.SPEECH_THRESHOLD) {
                // Speech detected
                if (!isSpeechDetected) {
                    isSpeechDetected = true;
                    speechStartTime = currentTime;
                    logDebug('Speech started', 'speech');
                }
                
                // Accumulate audio data
                audioBuffer.push(...inputData);

                // Update audio buffer display
                audioBufferDisplayDiv.textContent = `Audio Buffer: ${audioBuffer.length} samples`;
            } 
            else if (rms <= VAD_CONFIG.SILENCE_THRESHOLD && isSpeechDetected) {
                // Silence detected after speech
                const speechDuration = currentTime - speechStartTime;
                
                // Send audio if we have a buffer and it's not too long
                if (audioBuffer.length > 0 && speechDuration < VAD_CONFIG.MAX_RECORDING_DURATION) {
                    sendAudioToServer(new Float32Array(audioBuffer), nativeSampleRate);
                    logDebug(`Speech ended. Duration: ${speechDuration}ms`, 'warning');
                }

                // Reset state
                audioBuffer = [];
                audioBufferDisplayDiv.textContent = 'Audio Buffer: 0 samples';
                isSpeechDetected = false;
                speechStartTime = 0;
            }
        };

        logDebug('Advanced recording started', 'info');
        statusDiv.textContent = 'Listening for speech...';
        isRecording = true;

    } catch (error) {
        logDebug(`Initialization Error: ${error.message}`, 'error');
        statusDiv.textContent = `Error: ${error.message}`;
    }
}

// Start recording on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Comprehensive device and permission check
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia not supported');
    }

    // Advanced audio constraints
    const constraints = {
        audio: {
            echoCancellation: false,
            autoGainControl: false,
            noiseSuppression: false,
            channelCount: 1,
            latency: 0
        }
    };

    // Get audio stream
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    startAdvancedRecording();
});