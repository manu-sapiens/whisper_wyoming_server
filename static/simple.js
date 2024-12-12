// Server Configuration
const WHISPER_SERVER = {
    host: 'localhost',
    port: 8039
};

// DOM Elements
const statusDiv = document.getElementById('status');
const recordButton = document.getElementById('recordButton');
const transcriptionDiv = document.getElementById('transcriptionResult');

let mediaRecorder;
let audioChunks = [];

// Update status with different styles
function updateStatus(message, type = 'info') {
    if (statusDiv) {
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
    }
}

// Check Whisper server health
async function checkServerHealth() {
    try {
        const response = await fetch('/check_whisper_health');
        const data = await response.json();
        
        if (response.ok && data.status === 'ok') {
            updateStatus('Ready to record', 'success');
            return true;
        } else {
            updateStatus('Whisper server is not available', 'error');
            return false;
        }
    } catch (error) {
        updateStatus('Cannot connect to server', 'error');
        return false;
    }
}

// Initialize recording
async function initializeRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks);
            audioChunks = [];

            if (audioBlob.size > 0) {
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');

                try {
                    updateStatus('Transcribing...', 'info');
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
                    if (result.transcription) {
                        transcriptionDiv.textContent = result.transcription;
                        updateStatus('Ready to record', 'success');
                    }
                } catch (error) {
                    updateStatus('Error transcribing audio', 'error');
                }
            }
        };

        // Set up record button
        recordButton.addEventListener('mousedown', () => {
            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                audioChunks = [];
                mediaRecorder.start();
                recordButton.classList.add('recording');
                updateStatus('Recording...', 'info');
            }
        });

        recordButton.addEventListener('mouseup', () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                recordButton.classList.remove('recording');
                updateStatus('Processing...', 'info');
            }
        });

        updateStatus('Ready to record', 'success');
    } catch (error) {
        updateStatus('Microphone access denied', 'error');
        recordButton.disabled = true;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', async () => {
    await checkServerHealth();
    await initializeRecording();
});
