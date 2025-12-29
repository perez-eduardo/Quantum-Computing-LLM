// State
let isLoading = false;
let hasMessages = false;
let selectedModel = 'groq';
let dropdownOpen = false;
let isFirstQuestion = true; // Track cold start
let lastActivityTime = Date.now(); // Track idle time
const idleThreshold = 240000; // 4 minutes in ms (server sleeps at ~5 min)

// Pipeline steps for custom model
const PIPELINE_STEPS = [
    { text: "Embedding your question with Voyage AI", duration: 11 },
    { text: "Searching knowledge base (28,071 Q&A pairs)", duration: 11 },
    { text: "Retrieved relevant context from Neon database", duration: 11 },
    { text: "Ranking results by relevance", duration: 11 },
    { text: "Building prompt with context", duration: 11 },
    { text: "Loading 125.8M parameter model", duration: 11 },
    { text: "Generating response", duration: 60 }
];

// DOM elements
const chat = document.getElementById('chat');
const welcome = document.getElementById('welcome');
const messages = document.getElementById('messages');
const inputField = document.getElementById('input-field');
const sendBtn = document.getElementById('send-btn');
const inputLoader = document.getElementById('input-loader');
const loadingVideo = document.getElementById('loading-video');
const modelButton = document.getElementById('model-button');
const modelLabel = document.getElementById('model-label');
const modelDropdown = document.getElementById('model-dropdown');
const headerSubtitle = document.getElementById('header-subtitle');
const welcomeDisclaimer = document.getElementById('welcome-disclaimer');

// Modal functions
function openModal(modalId) {
    document.getElementById(modalId).classList.add('modal--open');
    document.body.style.overflow = 'hidden';
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('modal--open');
    document.body.style.overflow = '';
}

function closeModalOnBackdrop(event, modalId) {
    if (event.target.classList.contains('modal')) {
        closeModal(modalId);
    }
}

// Close modal on Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        document.querySelectorAll('.modal--open').forEach(modal => {
            modal.classList.remove('modal--open');
        });
        document.body.style.overflow = '';
    }
});

// Toggle dropdown
function toggleDropdown() {
    dropdownOpen = !dropdownOpen;
    modelDropdown.classList.toggle('model-selector__dropdown--open', dropdownOpen);
    modelButton.classList.toggle('model-selector__button--open', dropdownOpen);
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const selector = document.getElementById('model-selector');
    if (!selector.contains(event.target) && dropdownOpen) {
        dropdownOpen = false;
        modelDropdown.classList.remove('model-selector__dropdown--open');
        modelButton.classList.remove('model-selector__button--open');
    }
});

// Select model
function selectModel(model) {
    selectedModel = model;
    dropdownOpen = false;
    modelDropdown.classList.remove('model-selector__dropdown--open');
    modelButton.classList.remove('model-selector__button--open');
    updateUIForModel();
}

// Update UI elements based on selected model
function updateUIForModel() {
    if (selectedModel === 'groq') {
        modelLabel.textContent = 'Groq Model';
        headerSubtitle.textContent = 'Powered by Llama 3.3 70B via Groq';
        welcomeDisclaimer.style.display = 'none';
    } else {
        modelLabel.textContent = 'Custom Model';
        headerSubtitle.textContent = 'Powered by a custom 125.8M parameter transformer';
        welcomeDisclaimer.style.display = 'block';
    }
}

// Handle form submission
function handleSubmit(event) {
    event.preventDefault();
    const question = inputField.value.trim();
    if (question && !isLoading) {
        sendQuestion(question);
        inputField.value = '';
    }
}

// Hide any suggested buttons that match the question
function hideSuggestedButtons(question) {
    const buttons = document.querySelectorAll('.suggested__button');
    buttons.forEach(btn => {
        if (btn.textContent.trim().toLowerCase() === question.trim().toLowerCase()) {
            btn.parentElement.style.display = 'none';
        }
    });
}

// Send a question to the API
async function sendQuestion(question) {
    if (isLoading) return;

    hideSuggestedButtons(question);

    if (!hasMessages) {
        welcome.style.display = 'none';
        messages.style.display = 'flex';
        hasMessages = true;
    }

    addMessage('user', question);

    setLoading(true);
    
    // Check if server might be asleep (idle > 4 min, but not first question)
    const isIdle = !isFirstQuestion && (Date.now() - lastActivityTime) > idleThreshold;
    
    if (isIdle) {
        // Show wake-up message and ping health first
        const wakeUpEl = addWakeUpIndicator();
        
        const serverAwake = await waitForServerWakeUp();
        wakeUpEl.remove();
        
        if (!serverAwake) {
            addMessage('error', 'Server is taking too long to wake up. Please try again in a moment.');
            setLoading(false);
            return;
        }
    }
    
    // Now send the actual query
    const loadingEl = addLoadingIndicator();

    try {
        // Set timeout based on model (45s for Groq, 120s for Custom)
        const timeout = selectedModel === 'groq' ? 45000 : 120000;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                question,
                use_groq: selectedModel === 'groq'
            }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);

        const data = await response.json();

        loadingEl.remove();

        if (response.ok) {
            addMessage('ai', data.answer, data.response_time_ms, data.suggested_question, data.llm_used);
            isFirstQuestion = false;
            lastActivityTime = Date.now(); // Update activity time
        } else {
            addMessage('error', data.error || 'Something went wrong. Please try again.');
        }
    } catch (error) {
        loadingEl.remove();
        
        if (error.name === 'AbortError') {
            addMessage('error', 'Request timed out. The server may be busy. Please try again.');
        } else {
            addMessage('error', 'Failed to connect. Please try again.');
        }
    } finally {
        setLoading(false);
    }
}

// Wait for server to wake up by pinging health endpoint
async function waitForServerWakeUp() {
    const maxWaitTime = 30000; // 30 seconds max
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitTime) {
        try {
            const response = await Promise.race([
                fetch('/api/health', { method: 'GET' }),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('timeout')), 5000)
                )
            ]);
            
            if (response.ok) {
                lastActivityTime = Date.now();
                return true;
            }
        } catch (error) {
            // Timeout or error, wait a bit before retrying
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }
    
    return false;
}

// Add wake-up loading indicator
function addWakeUpIndicator() {
    const wakeUpDiv = document.createElement('div');
    wakeUpDiv.className = 'loading';
    wakeUpDiv.id = 'wakeup-indicator';
    wakeUpDiv.innerHTML = `
        <div class="loading__bubble">
            <span class="loading__message">Waking up server</span><span class="loading__dots-text" id="wakeup-dots"></span>
        </div>
    `;
    
    messages.appendChild(wakeUpDiv);
    scrollToBottom();
    
    showVideo();
    playVideo();
    
    // Start dots animation
    let dotsCount = 0;
    const dotsInterval = setInterval(() => {
        const dotsEl = document.getElementById('wakeup-dots');
        if (dotsEl) {
            dotsEl.textContent = '.'.repeat(dotsCount);
            dotsCount = (dotsCount + 1) % 4;
        }
    }, 400);
    
    // Store interval for cleanup
    wakeUpDiv.dotsInterval = dotsInterval;
    
    // Override remove to clear interval
    const originalRemove = wakeUpDiv.remove.bind(wakeUpDiv);
    wakeUpDiv.remove = function() {
        clearInterval(this.dotsInterval);
        originalRemove();
    };
    
    return wakeUpDiv;
}

// Add a message to the chat
function addMessage(type, content, responseTime = null, suggestedQuestion = null, llmUsed = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message--${type}`;

    if (type === 'ai') {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message__content';
        messageDiv.appendChild(contentDiv);

        let metaDiv = null;
        let suggestedDiv = null;

        if (responseTime) {
            metaDiv = document.createElement('div');
            metaDiv.className = 'message__meta';
            metaDiv.style.opacity = '0';
            
            let metaText = `Response time: ${(responseTime / 1000).toFixed(1)}s`;
            if (llmUsed) {
                const modelLabelText = llmUsed === 'groq' ? 'Groq' : 'Custom';
                metaText += ` · ${modelLabelText}`;
            }
            metaDiv.textContent = metaText;
            messageDiv.appendChild(metaDiv);
        }

        if (suggestedQuestion) {
            suggestedDiv = document.createElement('div');
            suggestedDiv.className = 'suggested';
            suggestedDiv.id = `suggested-${Date.now()}`;
            suggestedDiv.style.opacity = '0';
            
            const suggestedBtn = document.createElement('button');
            suggestedBtn.className = 'suggested__button';
            suggestedBtn.textContent = suggestedQuestion;
            suggestedBtn.addEventListener('click', function() {
                this.parentElement.style.display = 'none';
                sendQuestion(suggestedQuestion);
            });
            suggestedDiv.appendChild(suggestedBtn);
            messageDiv.appendChild(suggestedDiv);
        }

        messages.appendChild(messageDiv);
        scrollToBottom();

        typeText(contentDiv, content, () => {
            if (metaDiv) {
                metaDiv.style.transition = 'opacity 0.3s ease';
                metaDiv.style.opacity = '1';
            }
            if (suggestedDiv) {
                suggestedDiv.style.transition = 'opacity 0.3s ease';
                suggestedDiv.style.opacity = '1';
            }
            pauseVideo();
        });
    } else {
        let html = `<div class="message__content">${escapeHtml(content)}</div>`;
        messageDiv.innerHTML = html;
        messages.appendChild(messageDiv);
        scrollToBottom();
        
        if (type === 'error') {
            pauseVideo();
        }
    }
}

// Typing effect function
function typeText(element, text, onComplete) {
    let index = 0;
    const speed = 5; // milliseconds per character

    function type() {
        if (index < text.length) {
            element.textContent += text.charAt(index);
            index++;
            scrollToBottom();
            setTimeout(type, speed);
        } else {
            if (onComplete) onComplete();
        }
    }

    type();
}

// Add loading indicator
function addLoadingIndicator() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.id = 'loading-indicator';

    if (selectedModel === 'groq') {
        // Groq: just show "Thinking..."
        const firstQueryNote = isFirstQuestion ? '<div class="loading__patience"><div class="loading__first-query">Server is waking up from sleep.</div></div>' : '';
        loadingDiv.innerHTML = `
            <div class="loading__bubble" id="loading-bubble">
                <span class="loading__message" id="loading-message">Thinking</span><span class="loading__dots-text" id="loading-dots-text"></span>
            </div>
            ${firstQueryNote}
        `;
    } else {
        loadingDiv.innerHTML = `
            <div class="loading__bubble" id="loading-bubble">
                <span class="loading__message" id="loading-message">${PIPELINE_STEPS[0].text}</span><span class="loading__dots-text" id="loading-dots-text"></span>
            </div>
            <div class="loading__patience">
                This typically takes 40-90 seconds on our free-tier CPU server.
            </div>
        `;
    }

    messages.appendChild(loadingDiv);
    scrollToBottom();

    showVideo();
    playVideo();

    startLoadingAnimations();

    return loadingDiv;
}

// Show the video loader area
function showVideo() {
    inputLoader.classList.add('input-bar__loader--visible');
}

// Hide the video loader area
function hideVideo() {
    inputLoader.classList.remove('input-bar__loader--visible');
}

// Play video animation
function playVideo() {
    loadingVideo.play();
    loadingVideo.classList.remove('loading__video--paused');
}

// Pause video animation
function pauseVideo() {
    loadingVideo.pause();
    loadingVideo.classList.add('loading__video--paused');
}

// Loading animation intervals
let stepTimeouts = [];
let dotsInterval = null;

function startLoadingAnimations() {
    let dotsCount = 0;
    dotsInterval = setInterval(() => {
        const dotsEl = document.getElementById('loading-dots-text');
        if (dotsEl) {
            dotsEl.textContent = '.'.repeat(dotsCount);
            dotsCount = (dotsCount + 1) % 4;
        }
    }, 400);

    // Pipeline steps for custom model
    if (selectedModel === 'custom') {
        let elapsed = 0;
        
        PIPELINE_STEPS.forEach((step, index) => {
            if (index === 0) return;
            
            elapsed += PIPELINE_STEPS[index - 1].duration * 1000;
            
            const timeout = setTimeout(() => {
                const messageEl = document.getElementById('loading-message');
                const dotsEl = document.getElementById('loading-dots-text');
                
                if (messageEl) {
                    messageEl.style.opacity = '0';
                    if (dotsEl) dotsEl.style.opacity = '0';
                    
                    setTimeout(() => {
                        messageEl.textContent = step.text;
                        messageEl.style.opacity = '1';
                        if (dotsEl) dotsEl.style.opacity = '1';
                    }, 400);
                }
            }, elapsed);
            
            stepTimeouts.push(timeout);
        });
    }
}

function stopLoadingAnimations() {
    stepTimeouts.forEach(timeout => clearTimeout(timeout));
    stepTimeouts = [];
    if (dotsInterval) {
        clearInterval(dotsInterval);
        dotsInterval = null;
    }
}

// Set loading state
function setLoading(loading) {
    isLoading = loading;
    inputField.disabled = loading;
    sendBtn.disabled = loading;

    if (!loading) {
        stopLoadingAnimations();
    }
}

// Scroll chat to bottom
function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Backend availability check
const startupScreen = document.getElementById('startup');
const startupError = document.getElementById('startup-error');
const startupMessage = document.getElementById('startup-message');
const startupDots = document.getElementById('startup-dots');
const chatContainer = document.getElementById('chat');
const inputForm = document.getElementById('input-form');

let startupDotsInterval = null;
const maxCheckTime = 10000; // 10 seconds

function startStartupDots() {
    let count = 0;
    startupDotsInterval = setInterval(() => {
        startupDots.textContent = '.'.repeat(count);
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
    // Reset to loading state
    startupScreen.style.display = 'flex';
    startupError.style.display = 'none';
    
    startStartupDots();
    
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxCheckTime) {
        try {
            const response = await Promise.race([
                fetch('/api/health', { method: 'GET' }),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('timeout')), 3000)
                )
            ]);
            
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'ok' || data.groq_available) {
                    // Backend is ready
                    stopStartupDots();
                    startupScreen.style.display = 'none';
                    chatContainer.style.display = 'flex';
                    inputForm.style.display = 'flex';
                    lastActivityTime = Date.now(); // Server is awake now
                    return;
                }
            }
        } catch (error) {
            // Timeout or connection failed, wait before retry
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        // Update message based on progress
        const elapsed = Date.now() - startTime;
        if (elapsed > 6000) {
            startupMessage.textContent = 'Almost there';
        } else if (elapsed > 3000) {
            startupMessage.textContent = 'Waking up server';
        }
    }
    
    // Timeout - show error
    stopStartupDots();
    startupScreen.style.display = 'none';
    startupError.style.display = 'flex';
}

// Initialize UI on page load
updateUIForModel();
checkBackendAvailability();