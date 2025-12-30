// State
let isLoading = false;
let hasMessages = false;

// DOM elements
const chat = document.getElementById('chat');
const welcome = document.getElementById('welcome');
const messages = document.getElementById('messages');
const inputField = document.getElementById('input-field');
const sendBtn = document.getElementById('send-btn');
const inputLoader = document.getElementById('input-loader');
const loadingVideo = document.getElementById('loading-video');

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

document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        document.querySelectorAll('.modal--open').forEach(modal => {
            modal.classList.remove('modal--open');
        });
        document.body.style.overflow = '';
    }
});

// Handle form submission
function handleSubmit(event) {
    event.preventDefault();
    const question = inputField.value.trim();
    if (question && !isLoading) {
        sendQuestion(question);
        inputField.value = '';
    }
}

// Hide suggested buttons that match the question
function hideSuggestedButtons(question) {
    const buttons = document.querySelectorAll('.suggested__button');
    buttons.forEach(btn => {
        if (btn.textContent.trim().toLowerCase() === question.trim().toLowerCase()) {
            btn.parentElement.style.display = 'none';
        }
    });
}

// Send question to API
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
    const loadingEl = addLoadingIndicator();

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);
        
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        const data = await response.json();
        loadingEl.remove();

        if (response.ok) {
            addMessage('ai', data.answer, data.response_time_ms, data.suggested_question);
        } else {
            addMessage('error', data.error || 'Something went wrong.');
        }
    } catch (error) {
        loadingEl.remove();
        if (error.name === 'AbortError') {
            addMessage('error', 'Request timed out. Please try again.');
        } else {
            addMessage('error', 'Failed to connect. Please try again.');
        }
    } finally {
        setLoading(false);
    }
}

// Add message to chat
function addMessage(type, content, responseTime = null, suggestedQuestion = null) {
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
            metaDiv.textContent = `Response time: ${(responseTime / 1000).toFixed(1)}s`;
            messageDiv.appendChild(metaDiv);
        }

        if (suggestedQuestion) {
            suggestedDiv = document.createElement('div');
            suggestedDiv.className = 'suggested';
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
        messageDiv.innerHTML = `<div class="message__content">${escapeHtml(content)}</div>`;
        messages.appendChild(messageDiv);
        scrollToBottom();
        if (type === 'error') pauseVideo();
    }
}

// Typing effect
function typeText(element, text, onComplete) {
    let index = 0;
    const speed = 5;

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

// Loading indicator
let dotsInterval = null;

function addLoadingIndicator() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.innerHTML = `
        <div class="loading__bubble">
            <span class="loading__message">Thinking</span><span class="loading__dots-text" id="loading-dots-text"></span>
        </div>
    `;

    messages.appendChild(loadingDiv);
    scrollToBottom();
    showVideo();
    playVideo();
    startDotsAnimation();

    return loadingDiv;
}

function startDotsAnimation() {
    let dotsCount = 0;
    dotsInterval = setInterval(() => {
        const dotsEl = document.getElementById('loading-dots-text');
        if (dotsEl) {
            dotsEl.textContent = '.'.repeat(dotsCount);
            dotsCount = (dotsCount + 1) % 4;
        }
    }, 400);
}

function stopDotsAnimation() {
    if (dotsInterval) {
        clearInterval(dotsInterval);
        dotsInterval = null;
    }
}

function showVideo() {
    inputLoader.classList.add('input-bar__loader--visible');
}

function playVideo() {
    loadingVideo.play();
    loadingVideo.classList.remove('loading__video--paused');
}

function pauseVideo() {
    loadingVideo.pause();
    loadingVideo.classList.add('loading__video--paused');
}

function setLoading(loading) {
    isLoading = loading;
    inputField.disabled = loading;
    sendBtn.disabled = loading;
    if (!loading) stopDotsAnimation();
}

function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Startup check
const startupScreen = document.getElementById('startup');
const startupError = document.getElementById('startup-error');
const startupMessage = document.getElementById('startup-message');
const startupDots = document.getElementById('startup-dots');
const chatContainer = document.getElementById('chat');
const inputForm = document.getElementById('input-form');

let startupDotsInterval = null;

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
    startupScreen.style.display = 'flex';
    startupError.style.display = 'none';
    startStartupDots();
    
    const startTime = Date.now();
    const maxWait = 15000;
    
    while (Date.now() - startTime < maxWait) {
        try {
            const response = await Promise.race([
                fetch('/api/health', { method: 'GET' }),
                new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 5000))
            ]);
            
            if (response.ok) {
                stopStartupDots();
                startupScreen.style.display = 'none';
                chatContainer.style.display = 'flex';
                inputForm.style.display = 'flex';
                return;
            }
        } catch (error) {
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }
    
    stopStartupDots();
    startupScreen.style.display = 'none';
    startupError.style.display = 'flex';
}

checkBackendAvailability();
