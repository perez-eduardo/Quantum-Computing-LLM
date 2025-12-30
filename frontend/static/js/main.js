let isLoading = false;
let hasMessages = false;

const chat = document.getElementById('chat');
const welcome = document.getElementById('welcome');
const messages = document.getElementById('messages');
const inputField = document.getElementById('input-field');
const sendBtn = document.getElementById('send-btn');
const loadingVideo = document.getElementById('loading-video');

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

async function sendQuestion(question) {
    if (isLoading) return;
    hideSuggestedButtons(question);
    const isFirstMessage = !hasMessages;

    if (isFirstMessage) {
        hasMessages = true;
        playVideo();
    } else {
        welcome.style.display = 'none';
        messages.style.display = 'flex';
    }

    addMessage('user', question);
    setLoading(true);
    const loadingEl = isFirstMessage ? null : addLoadingIndicator();

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
        if (loadingEl) loadingEl.remove();
        
        // First message: transition from welcome to messages
        if (welcome.style.display !== 'none') {
            pauseVideo();
            welcome.style.display = 'none';
            messages.style.display = 'flex';
        }

        if (response.ok) {
            addMessage('ai', data.answer, data.response_time_ms, data.suggested_question);
        } else {
            addMessage('error', data.error || 'Something went wrong.');
        }
    } catch (error) {
        if (loadingEl) loadingEl.remove();
        
        // First message error: transition from welcome to messages
        if (welcome.style.display !== 'none') {
            pauseVideo();
            welcome.style.display = 'none';
            messages.style.display = 'flex';
        }
        
        addMessage('error', error.name === 'AbortError' ? 'Request timed out.' : 'Failed to connect.');
    } finally {
        setLoading(false);
    }
}

function addMessage(type, content, responseTime = null, suggestedQuestion = null) {
    const div = document.createElement('div');
    div.className = `message message--${type}`;

    if (type === 'ai') {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message__content';
        div.appendChild(contentDiv);

        let metaDiv = null, suggestedDiv = null;

        if (responseTime) {
            metaDiv = document.createElement('div');
            metaDiv.className = 'message__meta';
            metaDiv.style.opacity = '0';
            metaDiv.textContent = `Response time: ${(responseTime / 1000).toFixed(1)}s`;
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
    } else {
        div.innerHTML = `<div class="message__content">${escapeHtml(content)}</div>`;
        messages.appendChild(div);
        scrollToBottom();
        if (type === 'error') pauseVideo();
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

let dotsInterval = null;

function addLoadingIndicator() {
    const div = document.createElement('div');
    div.className = 'loading';
    div.innerHTML = `<div class="loading__bubble"><span class="loading__message">Thinking</span><span class="loading__dots"></span></div>`;
    messages.appendChild(div);
    scrollToBottom();
    playVideo();
    startDots();
    return div;
}

function startDots() {
    let count = 0;
    dotsInterval = setInterval(() => {
        const el = document.querySelector('.loading__dots');
        if (el) el.textContent = '.'.repeat(count = (count + 1) % 4);
    }, 400);
}

function stopDots() {
    if (dotsInterval) { clearInterval(dotsInterval); dotsInterval = null; }
}

function playVideo() { loadingVideo.play(); loadingVideo.classList.remove('loading__video--paused'); }
function pauseVideo() { loadingVideo.pause(); loadingVideo.classList.add('loading__video--paused'); }

function setLoading(loading) {
    isLoading = loading;
    inputField.disabled = loading;
    sendBtn.disabled = loading;
    if (!loading) stopDots();
}

function scrollToBottom() { chat.scrollTop = chat.scrollHeight; }

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Startup
const startupScreen = document.getElementById('startup');
const startupError = document.getElementById('startup-error');
const startupText = document.getElementById('startup-text');
const chatContainer = document.getElementById('chat');
const inputForm = document.getElementById('input-form');

let startupDotsInterval = null;

function startStartupDots() {
    let count = 0;
    startupDotsInterval = setInterval(() => {
        const base = 'Waking up server';
        startupText.textContent = base + '.'.repeat(count);
        count = (count + 1) % 4;
    }, 400);
}

function stopStartupDots() {
    if (startupDotsInterval) { clearInterval(startupDotsInterval); startupDotsInterval = null; }
}

async function checkBackendAvailability() {
    startupScreen.style.display = 'flex';
    startupError.style.display = 'none';
    startStartupDots();
    
    const start = Date.now();
    while (Date.now() - start < 15000) {
        try {
            const res = await Promise.race([
                fetch('/api/health'),
                new Promise((_, rej) => setTimeout(() => rej(), 5000))
            ]);
            if (res.ok) {
                stopStartupDots();
                startupScreen.style.display = 'none';
                chatContainer.style.display = 'flex';
                inputForm.style.display = 'flex';
                return;
            }
        } catch { await new Promise(r => setTimeout(r, 500)); }
    }
    
    stopStartupDots();
    startupScreen.style.display = 'none';
    startupError.style.display = 'flex';
}

checkBackendAvailability();