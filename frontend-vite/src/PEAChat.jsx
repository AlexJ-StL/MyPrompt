import React, { useState, useEffect } from 'react';

function PEAChat({ sessionId, initialMessage, onFinalize, selectedProvider, selectedModel }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

useEffect(() => {
    // Add the initial message from the /pea/start endpoint
    if (initialMessage) {
        setMessages([{ role: 'assistant', content: initialMessage }]);
    }
}, [initialMessage]);

const handleSendMessage = async () => {
    if (!input.trim() || loading || !sessionId) {
        return;
    }

    const newUserMessage = { role: 'user', content: input };
    setMessages(prevMessages => [...prevMessages, newUserMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
        const response = await fetch('/api/pea/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: sessionId,
            message: newUserMessage.content,
            provider: selectedProvider,
            model: selectedModel,
        }),
        });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'An error occurred during chat');
    }

    const data = await response.json();
    const peaResponse = { role: 'assistant', content: data.response };
    setMessages(prevMessages => [...prevMessages, peaResponse]);

    } catch (err) {
        setError(err.message);
        setMessages(prevMessages => [...prevMessages, { role: 'error', content: `Error: ${err.message}` }]);
    } finally {
        setLoading(false);
    }
};


const handleFinalizeClick = () => {
    if (onFinalize && sessionId && !loading) {
        onFinalize(sessionId);
    }
    };
    return (
    <div className="pea-chat-container">
        <h2>Prompt Engineering Assistant</h2>
        <div className="chat-history">
        {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'You' : 'PEA'}:</strong> {msg.content}
        </div>
        ))}
        {loading && <div className="chat-message loading">PEA is thinking...</div>}
        </div>
        <div className="chat-input">
        <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => { if (e.key === 'Enter') handleSendMessage(); }}
            placeholder="Type your response to PEA..."
            disabled={loading || !sessionId}
        />
        <button onClick={handleSendMessage} disabled={loading || !sessionId}>Send</button>
        <button onClick={handleFinalizeClick} disabled={loading || !sessionId}>Finalize Prompt (XML)</button>
        </div>
        {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
    );
}

export default PEAChat;
