// WebSocket Configuration
const socket = io('http://localhost:5000', {
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000
});

// Track connection state
let isSocketConnected = false;

// Custom VAD Configuration
const VAD_CONFIG = {
    SPEECH_THRESHOLD: 0.01,    // Minimum RMS to consider as speech
    SILENCE_THRESHOLD: 0.0005, // Ultra-low threshold for silence
    MAX_RECORDING_DURATION: 10000 // Maximum recording duration (ms)
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

// Explicit connection event handlers
socket.on('connect', () => {
    console.log('WebSocket Connected');
    isSocketConnected = true;
    whisperStatusDiv.textContent = 'WebSocket: Connected ';
    whisperStatusDiv.style.color = 'green';
    
    // Retry starting recording if it was previously blocked
    if (typeof startAdvancedRecording === 'function') {
        startAdvancedRecording();
    }
});

socket.on('connect_error', (error) => {
    console.error('WebSocket Connection Error:', error);
    isSocketConnected = false;
    whisperStatusDiv.textContent = 'WebSocket: Connection Error ';
    whisperStatusDiv.style.color = 'red';
});

socket.on('disconnect', (reason) => {
    console.log('WebSocket Disconnected:', reason);
    isSocketConnected = false;
    whisperStatusDiv.textContent = 'WebSocket: Disconnected ';
    whisperStatusDiv.style.color = 'orange';
});

socket.on('transcription', (data) => {
    console.log('Transcription received:', data);
    if (data && data.text) {
        // Append or replace based on is_final flag
        if (data.is_final) {
            transcriptionDiv.textContent = data.text;
        } else {
            // Append partial transcription
            transcriptionDiv.textContent += data.text;
        }
        
        logDebug(`Transcription: ${data.text}`, 'info');
        logDebug(`Is Final: ${data.is_final}, Language: ${data.language}, Confidence: ${data.confidence}`, 'info');
    }
});

socket.on('transcription_error', (error) => {
    console.error('Transcription error:', error);  
    transcriptionDiv.textContent = `Transcription Error: ${error.error || 'Unknown error'}`;
    logDebug(`Transcription Error: ${error.error}`, 'error');
});

socket.on('test_response', (data) => {
    console.log('Test Response Received:', data);
});

function sendAudioToServer(audioData, sampleRate) {
    try {
        // Convert audio data to base64
        const base64Audio = btoa(String.fromCharCode.apply(null, new Uint8Array(audioData)));
        
        // Send via WebSocket
        socket.emit('transcribe', {
            audio_data: base64Audio,
            sample_rate: sampleRate
        });
        
        logDebug(`Sending audio segment: ${audioData.length} samples`, 'info');
    } catch (error) {
        logDebug(`Audio send error: ${error.message}`, 'error');
    }
}

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

function float32ArrayToWav(float32Array, sampleRate) {
    const header = new DataView(new ArrayBuffer(44));
    header.setUint8(0, 'R'.charCodeAt(0));
    header.setUint8(1, 'I'.charCodeAt(0));
    header.setUint8(2, 'F'.charCodeAt(0));
    header.setUint8(3, 'F'.charCodeAt(0));
    header.setUint32(4, float32Array.length * 4 + 36, true);
    header.setUint8(8, 'W'.charCodeAt(0));
    header.setUint8(9, 'A'.charCodeAt(0));
    header.setUint8(10, 'V'.charCodeAt(0));
    header.setUint8(11, 'E'.charCodeAt(0));
    header.setUint8(12, 'f'.charCodeAt(0));
    header.setUint8(13, 'm'.charCodeAt(0));
    header.setUint8(14, 't'.charCodeAt(0));
    header.setUint8(15, ' '.charCodeAt(0));
    header.setUint32(16, 16, true);
    header.setUint16(20, 1, true);
    header.setUint16(22, 1, true);
    header.setUint32(24, sampleRate, true);
    header.setUint32(28, sampleRate * 4, true);
    header.setUint16(32, 4, true);
    header.setUint16(34, 16, true);
    header.setUint8(36, 'd'.charCodeAt(0));
    header.setUint8(37, 'a'.charCodeAt(0));
    header.setUint8(38, 't'.charCodeAt(0));
    header.setUint8(39, 'a'.charCodeAt(0));
    header.setUint32(40, float32Array.length * 4, true);

    const audioBuffer = new Uint8Array(header.byteLength + float32Array.length * 4);
    audioBuffer.set(new Uint8Array(header.buffer), 0);
    audioBuffer.set(new Uint8Array(float32Array.buffer), header.byteLength);

    return audioBuffer;
}

async function startAdvancedRecording() {
    try {
        // Check WebSocket connection
        const whisperReady = isSocketConnected;
        if (!whisperReady) {
            statusDiv.textContent = 'Cannot start recording: WebSocket not connected';
            logDebug('WebSocket not connected', 'error');
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
                    // Convert audio buffer to WAV
                    const wavBuffer = float32ArrayToWav(new Float32Array(audioBuffer), nativeSampleRate);
                    
                    // Convert WAV buffer to base64
                    const base64Audio = btoa(String.fromCharCode.apply(null, new Uint8Array(wavBuffer)));
                    
                    // Send via WebSocket
                    socket.emit('transcribe', {
                        audio_data: base64Audio,
                        sample_rate: nativeSampleRate
                    });
                    
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

document.addEventListener('DOMContentLoaded', async () => {
    try {
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
    } catch (error) {
        logDebug(`Initialization Error: ${error.message}`, 'error');
        statusDiv.textContent = `Error: ${error.message}`;
    }
});
