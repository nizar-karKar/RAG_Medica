import './style.css';
import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000/rag';
const API_HEALTH_URL = 'http://127.0.0.1:8000/health';

// DOM Elements
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatContainer = document.getElementById('chatContainer');
const sendBtn = document.getElementById('sendBtn');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const systemStatusIndicator = document.querySelector('.status-indicator');
const systemStatusText = document.querySelector('.system-status span');

// ==== 1. Chat Functionality ==== //

const createMessageElement = (content, isUser = false) => {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

  const avatarDiv = document.createElement('div');
  avatarDiv.className = 'avatar';
  avatarDiv.innerHTML = isUser
    ? `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`
    : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 4H8V8H4V16H8V20H16V16H20V8H16V4Z" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>`;

  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.innerHTML = content.replace(/\n/g, '<br>');

  messageDiv.appendChild(avatarDiv);
  messageDiv.appendChild(contentDiv);

  return messageDiv;
};

const showTypingIndicator = () => {
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant typing';
  wrapper.id = 'typingIndicator';
  
  wrapper.innerHTML = `
    <div class="avatar">
       <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 4H8V8H4V16H8V20H16V16H20V8H16V4Z" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>
    </div>
    <div class="message-content">
       <div class="typing-indicator">
          <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
       </div>
    </div>
  `;
  chatContainer.appendChild(wrapper);
  scrollToBottom();
};

const removeTypingIndicator = () => {
  const indicator = document.getElementById('typingIndicator');
  if (indicator) indicator.remove();
};

const scrollToBottom = () => {
  chatContainer.scrollTop = chatContainer.scrollHeight;
};

const handleChatSubmit = async (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;

  // Add user message to UI
  chatContainer.appendChild(createMessageElement(query, true));
  chatInput.value = '';
  scrollToBottom();
  
  // Disable input & show loading state
  sendBtn.disabled = true;
  chatInput.disabled = true;
  showTypingIndicator();

  try {
    const res = await axios.post(`${API_BASE_URL}/query`, { query });
    removeTypingIndicator();
    chatContainer.appendChild(createMessageElement(res.data.answer || "Processing complete."));
  } catch (err) {
    removeTypingIndicator();
    let errorText = "Sorry, I ran into an error connecting to the RAG backend.";
    if (err.response && err.response.data && err.response.data.detail) {
        errorText = `Error: ${err.response.data.detail}`;
    }
    const errorElem = createMessageElement(errorText);
    errorElem.querySelector('.message-content').style.color = 'var(--error)';
    chatContainer.appendChild(errorElem);
  } finally {
    sendBtn.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
    scrollToBottom();
  }
};

chatForm.addEventListener('submit', handleChatSubmit);

// ==== 2. Drag & Drop Upload Functionality ==== //

const handleFile = async (file) => {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    setUploadStatus('Only PDF files are supported.', 'error');
    return;
  }

  setUploadStatus('<div class="loader"></div> <span>Extracting text & Embedding into ChromaDB...</span>', 'loading');
  
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    setUploadStatus(`<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> <span>${res.data.message || 'Stored successfully!'}</span>`, 'success');
    
    // Auto-timeout success message to clean UI
    setTimeout(() => { uploadStatus.innerHTML = ''; }, 6000);
  } catch (err) {
    let msg = "Failed to upload file.";
    if(err.response?.data?.detail) msg = err.response.data.detail;
    setUploadStatus(`<span>Error: ${msg}</span>`, 'error');
  }
};

const setUploadStatus = (html, type) => {
  uploadStatus.innerHTML = `<div class="status-badge ${type}">${html}</div>`;
};

// Drag & drop event bindings
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
  dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
  dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length) handleFile(files[0]);
});

// Click to upload binding
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

// ==== 3. System Health Check ==== //

const checkSystemHealth = async () => {
  try {
    const res = await axios.get(API_HEALTH_URL);
    if (res.data.status === 'ok') {
      systemStatusIndicator.className = 'status-indicator online';
      systemStatusText.textContent = 'System Online & Ready';
    } else {
      throw new Error('Health check invalid');
    }
  } catch (err) {
    systemStatusIndicator.className = 'status-indicator offline';
    systemStatusText.textContent = 'System Offline - Backend Disconnected';
  }
};

checkSystemHealth();
setInterval(checkSystemHealth, 30000);

// ==== 4. Voice Query Functionality ==== //

const voiceBar      = document.getElementById('voiceBar');
const startVoiceBtn = document.getElementById('startVoiceBtn');
const stopVoiceBtn  = document.getElementById('stopVoiceBtn');
const voiceStatusText = document.getElementById('voiceStatusText');

let mediaRecorder = null;
let audioChunks   = [];
let recordingTimer = null;
let secondsElapsed = 0;

const setVoiceRecording = (isRecording) => {
  startVoiceBtn.disabled = isRecording;
  stopVoiceBtn.disabled  = !isRecording;
  voiceBar.classList.toggle('recording', isRecording);

  if (isRecording) {
    secondsElapsed = 0;
    voiceStatusText.innerHTML = `<div class="recording-dot"></div> Recording… <span id="recTimer">0s</span>`;
    recordingTimer = setInterval(() => {
      secondsElapsed++;
      const el = document.getElementById('recTimer');
      if (el) el.textContent = `${secondsElapsed}s`;
    }, 1000);
  } else {
    clearInterval(recordingTimer);
    voiceStatusText.innerHTML = `Press <strong>Start</strong> to record your query`;
  }
};

startVoiceBtn.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];

    // Prefer webm for broad browser support
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    mediaRecorder.addEventListener('dataavailable', (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    });

    mediaRecorder.start(100); // collect chunks every 100ms
    setVoiceRecording(true);
  } catch (err) {
    voiceStatusText.innerHTML = `<span style="color:var(--error)">Microphone access denied</span>`;
    console.error('Mic access error:', err);
  }
});

stopVoiceBtn.addEventListener('click', () => {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach(t => t.stop()); // release mic
  setVoiceRecording(false);

  mediaRecorder.addEventListener('stop', async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

    // Show in chat that voice is being processed
    chatContainer.appendChild(createMessageElement('🎙️ <em>Voice query sent…</em>', true));
    scrollToBottom();
    showTypingIndicator();

    const formData = new FormData();
    formData.append('audio', audioBlob, 'voice_query.webm');

    try {
      const res = await axios.post(`${API_BASE_URL}/voice-query`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      removeTypingIndicator();
      chatContainer.appendChild(createMessageElement(res.data.answer || 'No response received.'));
    } catch (err) {
      removeTypingIndicator();
      let errorText = 'Voice query failed — could not reach the backend.';
      if (err.response?.data?.detail) errorText = `Error: ${err.response.data.detail}`;
      const errorElem = createMessageElement(errorText);
      errorElem.querySelector('.message-content').style.color = 'var(--error)';
      chatContainer.appendChild(errorElem);
    } finally {
      scrollToBottom();
    }
  }, { once: true });
});
