// =============================================================================
// STATE
// =============================================================================
let isLoading = false;
let hasMessages = false;
let selectedModel = 'groq';
let dropdownOpen = false;
let lastQuestion = ''; // Store last question for retry

// Timer tracking for cold start detection
let lastActivityTime = Date.now();
const idleThreshold = 300000; // 5 minutes in ms (servers sleep after ~5 min)

// Pipeline steps for custom model loading animation
const PIPELINE_STEPS = [
    { text: "Embedding your question with Voyage AI", duration: 11 },
    { text: "Searching knowledge base (28,071 Q&A pairs)", duration: 11 },
    { text: "Retrieved relevant context from Neon database", duration: 11 },
    { text: "Ranking results by relevance", duration: 11 },
    { text: "Building prompt with context", duration: 11 },
    { text: "Loading 140M parameter model", duration: 11 },
    { text: "Generating response", duration: 60 }
];

// Animation intervals
let dotsInterval = null;
let pipelineInterval = null;
let pipelineStepIndex = 0;

// =============================================================================
// DOM ELEMENTS
// =============================================================================
const chat = document.getElementById('chat');
const welcome = document.getElementById('welcome');
const messages = document.getElementById('messages');
const inputField = document.getElementById('input-field');
const sendBtn = document.getElementById('send-btn');
const loadingVideo = document.getElementById('loading-video');
const customToast = document.getElementById('custom-toast');

// Track if toast has been shown
let toastShown = false;

// =============================================================================
// MODEL SELECTOR
// =============================================================================
function toggleModelDropdown(e) {
    if (e) e.stopPropagation();
    
    const dropdown = document.getElementById('model-dropdown');
    const chevron = document.getElementById('model-chevron');
    dropdownOpen = !dropdownOpen;
    
    if (dropdownOpen) {
        dropdown.classList.add('model-selector__dropdown--open');
        chevron.style.transform = 'rotate(180deg)';
    } else {
        dropdown.classList.remove('model-selector__dropdown--open');
        chevron.style.transform = 'rotate(0deg)';
    }
}

function selectModel(model) {
    selectedModel = model;
    
    const labelEl = document.getElementById('selected-model-label');
    const checkGroq = document.getElementById('check-groq');
    const checkCustom = document.getElementById('check-custom');
    
    if (model === 'groq') {
        labelEl.textContent = 'Groq';
        checkGroq.style.display = 'inline';
        checkCustom.style.display = 'none';
    } else {
        labelEl.textContent = 'Custom';
        checkGroq.style.display = 'none';
        checkCustom.style.display = 'inline';
        
        // Show toast only once per page load
        if (!toastShown) {
            showCustomToast();
            toastShown = true;
        }
    }
    
    toggleModelDropdown();
}

function showCustomToast() {
    customToast.classList.add('toast--visible');
    customToast.classList.remove('toast--fading');
    
    // Stay visible for 5 seconds, then fade out over 3 seconds
    setTimeout(() => {
        customToast.classList.remove('toast--visible');
        customToast.classList.add('toast--fading');
    }, 5000);
}

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    const selector = document.getElementById('model-selector');
    if (dropdownOpen && !selector.contains(e.target)) {
        toggleModelDropdown();
    }
});

// =============================================================================
// MODAL FUNCTIONS
// =============================================================================
function openModal(id) {
    document.getElementById(id).classList.add('modal--open');
    document.body.style.overflow = 'hidden';
}

function closeModal(id) {
    document.getElementById(id).classList.remove('modal--open');
    document.body.style.overflow = '';
}

function closeModalOnBackdrop(e, id) {
    if (e.target.classList.contains('modal')) closeModal(id);
}

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal--open').forEach(m => m.classList.remove('modal--open'));
        document.body.style.overflow = '';
    }
});

// =============================================================================
// FORM HANDLING
// =============================================================================
function handleSubmit(e) {
    e.preventDefault();
    const q = inputField.value.trim();
    if (q && !isLoading) {
        sendQuestion(q);
        inputField.value = '';
    }
}

function hideSuggestedButtons(q) {
    document.querySelectorAll('.suggested__button').forEach(btn => {
        if (btn.textContent.trim().toLowerCase() === q.trim().toLowerCase()) {
            btn.parentElement.style.display = 'none';
        }
    });
}

// =============================================================================
// RETRY FUNCTION
// =============================================================================
function retryLastQuestion() {
    if (lastQuestion && !isLoading) {
        // Remove the error message
        const errorMessages = document.querySelectorAll('.message--error');
        errorMessages.forEach(msg => msg.remove());
        
        // Retry the question
        sendQuestion(lastQuestion, true); // true = isRetry
    }
}

// =============================================================================
// BACKEND HEALTH CHECK
// =============================================================================
async function ensureBackendAwake() {
    const maxAttempts = 10;
    const retryDelay = 2000; // 2 seconds between retries
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const res = await fetch('/api/health', { signal: controller.signal });
            clearTimeout(timeoutId);
            
            if (res.ok) {
                lastActivityTime = Date.now();
                return { success: true };
            }
        } catch (e) {
            // Backend still waking up, wait and retry
            if (attempt < maxAttempts) {
                await new Promise(r => setTimeout(r, retryDelay));
            }
        }
    }
    return { 
        success: false, 
        error: 'Backend server did not respond after 10 attempts (~20 seconds). The server may be down or experiencing issues. Please try again in a few minutes.'
    };
}

// =============================================================================
// MAIN QUERY FUNCTION
// =============================================================================
async function sendQuestion(question, isRetry = false) {
    if (isLoading) return;
    
    // Store for retry
    lastQuestion = question;
    
    if (!isRetry) {
        hideSuggestedButtons(question);
    }

    if (!hasMessages) {
        welcome.style.display = 'none';
        messages.style.display = 'flex';
        hasMessages = true;
    }

    if (!isRetry) {
        addMessage('user', question);
    }
    
    setLoading(true);
    playVideo();
    
    // Always check backend health first
    const isIdle = (Date.now() - lastActivityTime) > idleThreshold;
    
    // Show appropriate loading indicator for health check
    let loadingEl = isIdle 
        ? addWakeUpLoadingIndicator() 
        : addHealthCheckLoadingIndicator();
    
    const healthResult = await ensureBackendAwake();
    
    if (!healthResult.success) {
        loadingEl.remove();
        addMessage('error', healthResult.error, true);
        setLoading(false);
        return;
    }
    
    // Health check passed, now show query loading indicator
    loadingEl.remove();
    loadingEl = selectedModel === 'groq' 
        ? addGroqLoadingIndicator() 
        : addCustomLoadingIndicator();
    
    // Determine timeout based on model
    let timeout;
    if (selectedModel === 'groq') {
        timeout = 30000; // 30 seconds for Groq
    } else {
        timeout = 180000; // 3 min for custom model
    }

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                question,
                model: selectedModel
            }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        const data = await response.json();
        loadingEl.remove();
        stopPipelineAnimation();

        if (response.ok) {
            // Update activity time on success
            lastActivityTime = Date.now();
            addMessage('ai', data.answer, data.response_time_ms, data.suggested_question, data.model_used);
        } else {
            addMessage('error', data.error || 'Something went wrong.', true);
        }
    } catch (error) {
        loadingEl.remove();
        stopPipelineAnimation();
        
        if (error.name === 'AbortError') {
            if (selectedModel === 'custom') {
                addMessage('error', 'Request timed out. The custom model may still be loading. Please try again.', true);
            } else {
                addMessage('error', 'Request timed out. Please try again.', true);
            }
        } else {
            addMessage('error', 'Failed to connect.', true);
        }
    } finally {
        setLoading(false);
    }
}

// =============================================================================
// LOADING INDICATORS
// =============================================================================
function addWakeUpLoadingIndicator() {
    const div = document.createElement('div');
    div.className = 'loading';
    div.innerHTML = `
        <div class="loading__bubble">
            <span class="loading__message">Waking up backend server</span>
            <span class="loading__dots"></span>
        </div>
        <div class="loading__subtext">
            <i class="fa-solid fa-clock"></i> Server was idle, cold start may take 10-15 seconds
        </div>
    `;
    messages.appendChild(div);
    scrollToBottom();
    startDots();
    return div;
}

function addHealthCheckLoadingIndicator() {
    const div = document.createElement('div');
    div.className = 'loading';
    div.innerHTML = `
        <div class="loading__bubble">
            <span class="loading__message">Connecting to backend</span>
            <span class="loading__dots"></span>
        </div>
    `;
    messages.appendChild(div);
    scrollToBottom();
    startDots();
    return div;
}

function addGroqLoadingIndicator() {
    const div = document.createElement('div');
    div.className = 'loading';
    div.innerHTML = `<div class="loading__bubble"><span class="loading__message">Thinking</span><span class="loading__dots"></span></div>`;
    messages.appendChild(div);
    scrollToBottom();
    startDots();
    return div;
}

function addCustomLoadingIndicator() {
    const div = document.createElement('div');
    div.className = 'loading loading--custom';
    div.innerHTML = `
        <div class="loading__bubble">
            <span class="loading__message" id="pipeline-message">${PIPELINE_STEPS[0].text}</span>
            <span class="loading__dots"></span>
        </div>
        <div class="loading__subtext">
            <i class="fa-solid fa-hourglass-half"></i> Custom model responses typically take 1-2 minutes
        </div>
    `;
    messages.appendChild(div);
    scrollToBottom();
    startDots();
    startPipelineAnimation();
    return div;
}

function startDots() {
    stopDots(); // Clear any existing interval first
    let count = 0;
    dotsInterval = setInterval(() => {
        const el = document.querySelector('.loading__dots');
        if (el) el.textContent = '.'.repeat(count = (count + 1) % 4);
    }, 400);
}

function stopDots() {
    if (dotsInterval) { 
        clearInterval(dotsInterval); 
        dotsInterval = null; 
    }
}

function startPipelineAnimation() {
    pipelineStepIndex = 0;
    let elapsed = 0;
    
    pipelineInterval = setInterval(() => {
        elapsed += 1000;
        
        // Find current step based on elapsed time
        let totalDuration = 0;
        for (let i = 0; i < PIPELINE_STEPS.length; i++) {
            totalDuration += PIPELINE_STEPS[i].duration * 1000;
            if (elapsed < totalDuration) {
                if (pipelineStepIndex !== i) {
                    pipelineStepIndex = i;
                    updatePipelineMessage(PIPELINE_STEPS[i].text);
                }
                break;
            }
        }
        
        // Loop back if we've gone through all steps
        if (elapsed >= totalDuration) {
            elapsed = 0;
            pipelineStepIndex = 0;
        }
    }, 1000);
}

function updatePipelineMessage(text) {
    const el = document.getElementById('pipeline-message');
    if (el) {
        el.style.opacity = '0';
        setTimeout(() => {
            el.textContent = text;
            el.style.opacity = '1';
        }, 200);
    }
}

function stopPipelineAnimation() {
    if (pipelineInterval) {
        clearInterval(pipelineInterval);
        pipelineInterval = null;
    }
}

// =============================================================================
// MESSAGE FUNCTIONS
// =============================================================================
function addMessage(type, content, responseTime = null, suggestedQuestion = null, modelUsed = null) {
    const div = document.createElement('div');
    div.className = `message message--${type}`;

    if (type === 'ai') {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message__content';
        div.appendChild(contentDiv);

        let metaDiv = null, suggestedDiv = null;

        if (responseTime || modelUsed) {
            metaDiv = document.createElement('div');
            metaDiv.className = 'message__meta';
            metaDiv.style.opacity = '0';
            
            let metaText = '';
            if (responseTime) {
                metaText += `${(responseTime / 1000).toFixed(1)}s`;
            }
            if (modelUsed) {
                metaText += metaText ? ` · ${modelUsed}` : modelUsed;
            }
            metaDiv.textContent = metaText;
            div.appendChild(metaDiv);
        }

        if (suggestedQuestion) {
            suggestedDiv = document.createElement('div');
            suggestedDiv.className = 'suggested';
            suggestedDiv.style.opacity = '0';
            const btn = document.createElement('button');
            btn.className = 'suggested__button';
            btn.textContent = suggestedQuestion;
            btn.onclick = () => { suggestedDiv.style.display = 'none'; sendQuestion(suggestedQuestion); };
            suggestedDiv.appendChild(btn);
            div.appendChild(suggestedDiv);
        }

        messages.appendChild(div);
        scrollToBottom();

        typeText(contentDiv, content, () => {
            if (metaDiv) { metaDiv.style.transition = 'opacity 0.3s'; metaDiv.style.opacity = '1'; }
            if (suggestedDiv) { suggestedDiv.style.transition = 'opacity 0.3s'; suggestedDiv.style.opacity = '1'; }
            pauseVideo();
        });
    } else if (type === 'error') {
        // Check if showRetry is passed (3rd param for error type)
        const showRetry = responseTime === true;
        
        let innerHTML = `<div class="message__content">${escapeHtml(content)}</div>`;
        
        if (showRetry && lastQuestion) {
            innerHTML += `
                <div class="message__retry">
                    <button class="retry__button" onclick="retryLastQuestion()">
                        <i class="fa-solid fa-rotate-right"></i> Retry
                    </button>
                </div>
            `;
        }
        
        div.innerHTML = innerHTML;
        messages.appendChild(div);
        scrollToBottom();
        pauseVideo();
    } else {
        div.innerHTML = `<div class="message__content">${escapeHtml(content)}</div>`;
        messages.appendChild(div);
        scrollToBottom();
    }
}

function typeText(el, text, cb) {
    let i = 0;
    (function type() {
        if (i < text.length) {
            el.textContent += text.charAt(i++);
            scrollToBottom();
            setTimeout(type, 5);
        } else if (cb) cb();
    })();
}

// =============================================================================
// VIDEO CONTROLS
// =============================================================================
function playVideo() { 
    loadingVideo.play(); 
    loadingVideo.classList.remove('loading__video--paused'); 
}

function pauseVideo() { 
    loadingVideo.pause(); 
    loadingVideo.classList.add('loading__video--paused'); 
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================
function setLoading(loading) {
    isLoading = loading;
    inputField.disabled = loading;
    sendBtn.disabled = loading;
    if (!loading) {
        stopDots();
        stopPipelineAnimation();
    }
}

function scrollToBottom() { 
    chat.scrollTop = chat.scrollHeight; 
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// =============================================================================
// STARTUP / BACKEND AVAILABILITY CHECK
// =============================================================================
const startupScreen = document.getElementById('startup');
const startupError = document.getElementById('startup-error');
const startupText = document.getElementById('startup-text');
const chatContainer = document.getElementById('chat');
const inputForm = document.getElementById('input-form');

let startupDotsInterval = null;
const maxCheckTime = 30000; // 30 seconds for cold start

function startStartupDots() {
    let count = 0;
    startupDotsInterval = setInterval(() => {
        const base = 'Waking up server';
        startupText.textContent = base + '.'.repeat(count);
        count = (count + 1) % 4;
    }, 400);
}

function stopStartupDots() {
    if (startupDotsInterval) { 
        clearInterval(startupDotsInterval); 
        startupDotsInterval = null; 
    }
}

async function checkBackendAvailability() {
    startupScreen.style.display = 'flex';
    startupError.style.display = 'none';
    startStartupDots();
    
    const start = Date.now();
    while (Date.now() - start < maxCheckTime) {
        try {
            const res = await Promise.race([
                fetch('/api/health'),
                new Promise((_, rej) => setTimeout(() => rej(), 10000))
            ]);
            if (res.ok) {
                stopStartupDots();
                startupScreen.style.display = 'none';
                chatContainer.style.display = 'flex';
                inputForm.style.display = 'flex';
                lastActivityTime = Date.now(); // Server is awake
                return;
            }
        } catch { 
            await new Promise(r => setTimeout(r, 500)); 
        }
    }
    
    stopStartupDots();
    startupScreen.style.display = 'none';
    startupError.style.display = 'flex';
}

// =============================================================================
// INITIALIZE
// =============================================================================
checkBackendAvailability();