// script.js
document.addEventListener('DOMContentLoaded', () => {
    const pdfUpload = document.getElementById('pdf-upload');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const suggestedQuestions = document.getElementById('suggested-questions');
    let isProcessing = false;
    let debounceTimeout;

    // Initialize WebSocket for real-time updates
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    ws.onmessage = handleWebSocketMessage;

    function handleWebSocketMessage(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'processing_update') {
            updateProcessingStatus(data.message);
        } else if (data.type === 'suggested_questions') {
            updateSuggestedQuestions(data.questions);
        }
    }

    function updateProcessingStatus(message) {
        const statusDiv = document.getElementById('processing-status');
        if (statusDiv) {
            statusDiv.textContent = message;
        }
    }

    function updateSuggestedQuestions(questions) {
        if (suggestedQuestions) {
            suggestedQuestions.innerHTML = '';
            questions.forEach(question => {
                const btn = document.createElement('button');
                btn.className = 'suggested-question-btn';
                btn.textContent = question;
                btn.onclick = () => {
                    messageInput.value = question;
                    sendMessage();
                };
                suggestedQuestions.appendChild(btn);
            });
        }
    }

    async function uploadPDF(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            disableInterface(true);
            addMessage('system', 'Processing document...');

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const data = await response.json();
            addMessage('system', 'Document processed successfully!');

            if (data.suggested_questions) {
                updateSuggestedQuestions(data.suggested_questions);
            }

        } catch (error) {
            handleError(error);
        } finally {
            disableInterface(false);
        }
    }

    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message || isProcessing) return;

        clearTimeout(debounceTimeout);
        debounceTimeout = setTimeout(async () => {
            try {
                isProcessing = true;
                disableInterface(true);

                messageInput.value = '';
                const userMessageDiv = addMessage('user', message);
                const botTypingDiv = addMessage('bot', 'Thinking...');

                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 30000);

                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: message }),
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) throw new Error('Network response was not ok');

                const data = await response.json();
                if (data.error) throw new Error(data.error);

                botTypingDiv.innerHTML = formatMessage(data.response);
                botTypingDiv.classList.remove('typing');

                if (data.suggested_questions) {
                    updateSuggestedQuestions(data.suggested_questions);
                }

            } catch (error) {
                handleError(error);
            } finally {
                isProcessing = false;
                disableInterface(false);
            }
        }, 300);
    }

    function formatMessage(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>')
            .replace(/(`{3})(.*?)\1/gs, '<pre><code>$2</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');
    }

    function addMessage(type, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.innerHTML = type === 'user' ? text : formatMessage(text);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageDiv;
    }

    function disableInterface(disabled) {
        messageInput.disabled = disabled;
        sendButton.disabled = disabled;
        pdfUpload.disabled = disabled;
        
        if (disabled) {
            sendButton.classList.add('disabled');
        } else {
            sendButton.classList.remove('disabled');
        }
    }

    function handleError(error) {
        addMessage('system', `Error: ${error.message || 'Something went wrong. Please try again.'}`);
        console.error('Error:', error);
    }

    // Event Listeners
    pdfUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) uploadPDF(file);
    });

    sendButton.addEventListener('click', sendMessage);

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && ws.readyState === WebSocket.OPEN) {
            ws.close();
        } else if (!document.hidden && ws.readyState === WebSocket.CLOSED) {
            ws.connect();
        }
    });
});