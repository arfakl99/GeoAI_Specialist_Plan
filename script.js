// ========================================
// SAM2 Geospatial Segmentation - JavaScript
// ========================================

// Configuration
const API_URL = 'http://127.0.0.1:5050/segment/text';
let selectedFile = null;
let lastResponse = null;

// ========================================
// DOM Elements
// ========================================

const fileUploadArea = document.getElementById('fileUploadArea');
const fileInput = document.getElementById('fileInput');
const fileNameDisplay = document.getElementById('fileName');
const promptInput = document.getElementById('promptInput');
const modelIdInput = document.getElementById('modelId');
const backendSelect = document.getElementById('backend');
const outputFormatSelect = document.getElementById('outputFormat');
const confidenceThreshold = document.getElementById('confidenceThreshold');
const confidenceValue = document.getElementById('confidenceValue');
const minSizeInput = document.getElementById('minSize');
const maxSizeInput = document.getElementById('maxSize');
const detectionForm = document.getElementById('detectionForm');
const submitBtn = document.getElementById('submitBtn');
const resultsContainer = document.getElementById('resultsContainer');
const imageDisplay = document.getElementById('imageDisplay');
const resultImage = document.getElementById('resultImage');
const downloadSection = document.getElementById('downloadSection');
const downloadBtn = document.getElementById('downloadBtn');
const statusMessage = document.getElementById('statusMessage');
const statistics = document.getElementById('statistics');

// ========================================
// Event Listeners - File Upload
// ========================================

fileUploadArea.addEventListener('click', () => fileInput.click());

fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('drag-over');
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('drag-over');
});

fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove('drag-over');
    handleFileSelect(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
    }
});

// ========================================
// File Upload Handling
// ========================================

function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/tiff', 'image/png', 'image/jpeg', 'image/jpg'];
    const validExtensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg'];
    
    const isValidType = validTypes.includes(file.type) || 
                       validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

    if (!isValidType) {
        showStatus('❌ Please upload a valid image file (TIF, PNG, JPEG)', 'error');
        return;
    }

    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
        showStatus('❌ File size must be less than 50MB', 'error');
        return;
    }

    selectedFile = file;
    fileNameDisplay.textContent = `✅ File selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`;
    fileNameDisplay.classList.add('active');
    showStatus(`✅ File loaded: ${file.name}`, 'success');
}

// ========================================
// Prompt Suggestions
// ========================================

document.querySelectorAll('.suggestion-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        const prompt = btn.getAttribute('data-prompt');
        promptInput.value = prompt;
        promptInput.focus();
    });
});

// ========================================
// Confidence Threshold Slider
// ========================================

confidenceThreshold.addEventListener('input', (e) => {
    confidenceValue.textContent = e.target.value;
});

// ========================================
// Form Submission
// ========================================

detectionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Validate
    if (!selectedFile) {
        showStatus('❌ Please select an image file', 'error');
        return;
    }

    if (!promptInput.value.trim()) {
        showStatus('❌ Please enter a detection prompt', 'error');
        return;
    }

    // Run detection
    await runDetection();
});

// ========================================
// Run Detection
// ========================================

async function runDetection() {
    try {
        // Show loading state
        submitBtn.disabled = true;
        document.querySelector('.btn-text').style.display = 'none';
        document.querySelector('.btn-loader').style.display = 'inline-flex';
        showStatus('🔍 Running detection... This may take a moment', 'info');
        hideResults();

        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('prompt', promptInput.value.trim());
        formData.append('model_id', modelIdInput.value);
        formData.append('backend', backendSelect.value);
        formData.append('output_format', outputFormatSelect.value);
        formData.append('confidence_threshold', parseFloat(confidenceThreshold.value));
        formData.append('min_size', parseInt(minSizeInput.value) || 0);
        formData.append('max_size', parseInt(maxSizeInput.value) || 0);

        // Make API request
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
            mode: 'cors',
            credentials: 'omit'
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `API Error: ${response.status}`);
        }

        // Handle response based on output format
        const outputFormat = outputFormatSelect.value;

        if (outputFormat === 'json' || outputFormat === 'geojson') {
            // JSON response
            const jsonData = await response.json();
            lastResponse = jsonData;
            displayJsonResults(jsonData);
            showStatus('✅ Detection completed successfully!', 'success');
        } else {
            // Image response (PNG, TIF, GeoTIFF)
            const blob = await response.blob();
            displayImageResults(blob);
            showStatus('✅ Detection completed! Image generated.', 'success');
        }

    } catch (error) {
        console.error('Detection error:', error);
        showStatus(`❌ Error: ${error.message}`, 'error');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        document.querySelector('.btn-text').style.display = 'inline-flex';
        document.querySelector('.btn-loader').style.display = 'none';
    }
}

// ========================================
// Display JSON Results
// ========================================

function displayJsonResults(jsonData) {
    resultsContainer.innerHTML = '';
    imageDisplay.style.display = 'none';
    downloadSection.style.display = 'none';
    statistics.style.display = 'none';

    // Create result display
    const resultDiv = document.createElement('div');
    resultDiv.className = 'json-results';
    resultDiv.innerHTML = `
        <div style="flex: 1; overflow-y: auto; padding: 24px;">
            <h3 style="margin-top: 0; color: var(--primary-color); margin-bottom: 12px;">
                📊 Detection Results
            </h3>
            <pre style="background: var(--bg-secondary); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px;">
${JSON.stringify(jsonData, null, 2).substring(0, 2000)}${JSON.stringify(jsonData, null, 2).length > 2000 ? '\n...' : ''}
            </pre>
            <button id="copyJsonBtn" class="btn btn-secondary" style="margin-top: 12px;">
                📋 Copy JSON
            </button>
        </div>
    `;
    
    resultsContainer.appendChild(resultDiv);
    resultsContainer.style.display = 'block';

    // Copy JSON button
    document.getElementById('copyJsonBtn')?.addEventListener('click', () => {
        navigator.clipboard.writeText(JSON.stringify(jsonData, null, 2));
        showStatus('✅ JSON copied to clipboard!', 'success');
    });

    // Show download section
    downloadSection.style.display = 'flex';
    setupJsonDownload(jsonData);

    // Display statistics if available
    if (jsonData.num_detections !== undefined) {
        displayStatistics(jsonData);
    }
}

// ========================================
// Display Image Results
// ========================================

function displayImageResults(blob) {
    resultsContainer.innerHTML = '';
    statistics.style.display = 'none';

    // Create image URL
    const imageUrl = URL.createObjectURL(blob);
    resultImage.src = imageUrl;
    imageDisplay.style.display = 'flex';
    resultsContainer.style.display = 'block';

    // Show download section
    downloadSection.style.display = 'flex';
    setupImageDownload(blob);

    lastResponse = { blob, imageUrl };
}

// ========================================
// Display Statistics
// ========================================

function displayStatistics(data) {
    if (data.num_detections === undefined) return;

    const detections = data.detections || [];
    const highConf = detections.filter(d => d.score >= 0.8).length;
    const mediumConf = detections.filter(d => d.score >= 0.7 && d.score < 0.8).length;
    const lowConf = detections.filter(d => d.score < 0.7).length;

    document.getElementById('totalDetected').textContent = data.num_detections;
    document.getElementById('highConf').textContent = highConf;
    document.getElementById('mediumConf').textContent = mediumConf;
    document.getElementById('lowConf').textContent = lowConf;

    statistics.style.display = 'block';
}

// ========================================
// Download Functionality
// ========================================

function setupImageDownload(blob) {
    downloadBtn.innerHTML = '⬇️ Download Image';
    downloadBtn.onclick = () => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const extension = outputFormatSelect.value === 'tif' ? 'tif' : 'png';
        a.href = url;
        a.download = `detection_${promptInput.value}_${timestamp}.${extension}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };
}

function setupJsonDownload(jsonData) {
    downloadBtn.innerHTML = '⬇️ Download JSON';
    downloadBtn.onclick = () => {
        const dataStr = JSON.stringify(jsonData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const a = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        a.href = url;
        a.download = `detection_${promptInput.value}_${timestamp}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };
}

// ========================================
// UI Helpers
// ========================================

function showStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    statusMessage.style.display = 'block';

    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            statusMessage.style.display = 'none';
        }, 5000);
    }
}

function hideResults() {
    resultsContainer.innerHTML = `
        <div class="empty-state">
            <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                <path d="M21 15l-5-5L5 21"></path>
            </svg>
            <p class="empty-text">Processing your request...</p>
        </div>
    `;
    resultsContainer.style.display = 'block';
    imageDisplay.style.display = 'none';
    downloadSection.style.display = 'none';
    statistics.style.display = 'none';
}

// ========================================
// API Health Check
// ========================================

async function checkApiHealth() {
    try {
        const response = await fetch('http://127.0.0.1:5050/health', {
            mode: 'cors',
            credentials: 'omit'
        });
        
        if (response.ok) {
            console.log('✅ API is running');
            return true;
        } else {
            showStatus('⚠️ API connection issue', 'warning');
            return false;
        }
    } catch (error) {
        console.error('API health check failed:', error);
        showStatus('❌ Cannot connect to API. Make sure SAM2 API is running on http://127.0.0.1:5050', 'error');
        return false;
    }
}

// ========================================
// Initialize
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('SAM2 Web Application loaded');
    checkApiHealth();
});
