// WebSocket Configuration
const socket = io('http://localhost:5000', {
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000
});

// Track connection state
let isSocketConnected = false;

// Audio Processing Configuration
const VAD_CONFIG = {
    SPEECH_THRESHOLD: 0.01,    // Minimum RMS to consider as speech
    SILENCE_THRESHOLD: 0.0005, // Ultra-low threshold for silence
    MAX_RECORDING_DURATION: 10000, // Maximum recording duration (ms)
    SAMPLE_BUFFER_SIZE: 4096,  // Standard buffer size
    MAX_AUDIO_DURATION: 10     // Max audio duration in seconds
};

// Audio Processing State
let audioContext;
let mediaStream;
let audioWorkletNode;
let isRecording = false;
let peakRMS = 0;
let audioBuffer = [];
let isSpeechDetected = false;
let speechStartTime = 0;

// DOM Elements
const statusDiv = document.getElementById('status');
const debugDiv = document.getElementById('debug');
const transcriptionDiv = document.getElementById('transcriptionResult');
const rmsDisplayDiv = document.getElementById('rmsDisplay');
const audioBufferDisplayDiv = document.getElementById('audioBufferDisplay');
const whisperStatusDiv = document.getElementById('whisperStatus');

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
        if (!isSocketConnected) {
            statusDiv.textContent = 'Cannot start recording: WebSocket not connected';
            logDebug('WebSocket not connected', 'error');
            return;
        }

        // Create audio context with native sample rate
        audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Dynamically load AudioWorklet module with error handling
        try {
            const processorPath = window.AUDIO_PROCESSOR_PATH || '/static/audio-processor.js';
            await audioContext.audioWorklet.addModule(processorPath);
        } catch (moduleError) {
            console.error('AudioWorklet module loading error:', moduleError);
            statusDiv.textContent = `Module Load Error: ${moduleError.message}`;
            return;
        }

        // Create AudioWorklet node
        audioWorkletNode = new AudioWorkletNode(audioContext, 'whisper-audio-processor');

        // Setup message handler for the AudioWorklet
        audioWorkletNode.port.onmessage = (event) => {
            switch (event.data.type) {
                case 'speech_start':
                    logDebug('Speech started', 'speech');
                    rmsDisplayDiv.textContent = `Current RMS: ${event.data.rms.toFixed(4)}`;
                    break;
                
                case 'speech_end':
                    try {
                        const audioData = event.data.audioData;
                        const speechDuration = event.data.speechDuration;

                        // Convert audio data to WAV
                        const wavBuffer = float32ArrayToWav(audioData, audioContext.sampleRate);
                        
                        // Convert WAV buffer to base64
                        const base64Audio = btoa(String.fromCharCode.apply(null, new Uint8Array(wavBuffer)));
                        
                        // Send via WebSocket
                        socket.emit('transcribe', {
                            audio_data: base64Audio,
                            sample_rate: audioContext.sampleRate
                        });
                        
                        logDebug(`Speech ended. Duration: ${speechDuration}ms`, 'warning');
                    } catch (error) {
                        console.error('Audio transmission error:', error);
                    }
                    break;
            }
        };

        // Connect audio graph
        const source = audioContext.createMediaStreamSource(mediaStream);
        source.connect(audioWorkletNode);
        audioWorkletNode.connect(audioContext.destination);

        logDebug('Advanced recording started', 'info');
        statusDiv.textContent = 'Listening for speech...';
        isRecording = true;

    } catch (error) {
        logDebug(`Initialization Error: ${error.message}`, 'error');
        statusDiv.textContent = `Error: ${error.message}`;
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

// Start recording on page load
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
