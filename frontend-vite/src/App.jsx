import React, { useState } from 'react';
import PEAChat from './PEAChat';
import './App.css';

function App() {
  const [mode, setMode] = useState('standard'); // 'standard' or 'pea'
  const [userRequest, setUserRequest] = useState('');
  const [optimizedPrompt, setOptimizedPrompt] = useState('');
  const [peaSessionId, setPeaSessionId] = useState(null);
  const [peaInitialMessage, setPeaInitialMessage] = useState(null);
  const [finalXmlPrompt, setFinalXmlPrompt] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedProvider, setSelectedProvider] = useState('google'); // Default provider
  const [selectedModel, setSelectedModel] = useState(''); // Optional: User-selected model

  const instructions = {
    standard: {
      title: "Standard Prompt Mode Instructions:",
      points: [
        "Enter your complete prompt directly.",
        "Clearly state your goal or desired output.",
        "Provide necessary context or background information.",
        "Specify the desired output format (e.g., Markdown, code, etc.).",
        "Mention any important constraints or rules.",
        "If applicable, suggest a persona for the LLM.",
      ],
    },
    pea: {
      title: "PEA Mode Instructions:",
      points: [
        "This mode is best suited for refining prompts for complex, detailed, or novel tasks; therefore, if you are not sure if this mode is the right choice, please use the standard prompt mode.",
        "Start by entering your initial idea or task description for the prompt you want to build.",
        "Provide as much detail as possible in your initial request to significantly reduce the conversation length and token usage.",
        "The PEA will guide you by asking clarifying questions based on key areas like goals, audience, requirements, and constraints.",
        "Each message exchange with the PEA consumes tokens. Being concise and providing clear answers is recommended.",
        "Click the 'Finalize Prompt (XML)' button when you believe enough information has been provided to generate the complete prompt.",
      ],
    },
  };

  const handleStandardOptimizeClick = async () => {
    setLoading(true);
    setError(null);
    setOptimizedPrompt('');
    setFinalXmlPrompt(null); // Clear previous PEA output

    try {
      const response = await fetch('/api/optimize-prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          request: userRequest,
          provider: selectedProvider,
          model: selectedModel,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'An error occurred');
      }

      const data = await response.json();
      setOptimizedPrompt(data.optimized_prompt);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStartPeaSession = async () => {
    if (!userRequest.trim()) {
      setError("Please enter an initial request to start the PEA session.");
      return;
    }

    setLoading(true);
    setError(null);
    setOptimizedPrompt(''); // Clear standard output
    setFinalXmlPrompt(null); // Clear previous PEA output
    setPeaSessionId(null);
    setPeaInitialMessage(null);

    try {
      const response = await fetch('/api/pea/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ initial_request: userRequest }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'An error occurred starting PEA session');
      }

      const data = await response.json();
      setPeaSessionId(data.session_id);
      setPeaInitialMessage(data.response);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFinalizePrompt = async (sessionId) => {
    setLoading(true);
    setError(null);
    setFinalXmlPrompt(null); // Clear previous attempt

    try {
      const response = await fetch('/api/pea/finalize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'An error occurred finalizing prompt');
      }

      const data = await response.json();
      setFinalXmlPrompt(data.final_prompt);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      // Clear PEA session state after finalizing
      setPeaSessionId(null);
      setPeaInitialMessage(null);
      setUserRequest(''); // Clear the input field after finalization
    }
  };

  return (
    <div className="App">
      <h1>MyPrompt</h1>

      {/* Mode Selection */}
      <div>
        <button onClick={() => setMode('standard')} disabled={mode === 'standard' || loading}>Standard Mode</button>
        <button onClick={() => setMode('pea')} disabled={mode === 'pea' || loading}>PEA Mode</button>
      </div>

      {/* Instructions */}
      <div className="instructions">
        <h2>{instructions[mode].title}</h2>
        <ul>
          {instructions[mode].points.map((point, index) => (
            <li key={index}>{point}</li>
          ))}
        </ul>
      </div>

      {/* Input Area (Shown in Standard mode or before PEA session starts) */}
      {(mode === 'standard' || (mode === 'pea' && !peaSessionId)) && (
        <>
          <textarea
            rows="10"
            cols="50"
            value={userRequest}
            onChange={(e) => setUserRequest(e.target.value)}
            placeholder={mode === 'standard' ? "Enter your complete prompt here..." : "Enter your initial natural language request here to start PEA..."}
            disabled={loading}
          />
          {mode === 'standard' && (
            <div className="provider-model-selection">
              <label htmlFor="provider-select">Provider:</label>
              <select
                id="provider-select"
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                disabled={loading}
              >
                <option value="google">Google</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="openrouter">OpenRouter</option>
                <option value="groq">Groq</option>
                <option value="mistral">Mistral</option>
                <option value="ollama">Ollama</option>
                <option value="lmstudio">LM Studio</option>
              </select>

              <label htmlFor="model-input">Model (optional):</label>
              <input
                id="model-input"
                type="text"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                placeholder="e.g., gpt-4-turbo, claude-3-opus-20240229"
                disabled={loading}
              />
            </div>
          )}
        </>
      )}

      {/* Action Button (Changes based on mode and PEA session state) */}
      <br />
      {mode === 'standard' && (
        <button onClick={handleStandardOptimizeClick} disabled={loading}>
          {loading ? 'Optimizing...' : 'Optimize Prompt'}
        </button>
      )}
      {mode === 'pea' && !peaSessionId && (
        <button onClick={handleStartPeaSession} disabled={loading}>
          {loading ? 'Starting PEA...' : 'Start PEA Session'}
        </button>
      )}
      {mode === 'pea' && peaSessionId && (
        <button onClick={() => {
          setPeaSessionId(null);
          setPeaInitialMessage(null);
          setUserRequest('');
          setError(null); // Clear any error messages
          setLoading(false); // Ensure loading is false
        }} disabled={loading}>Cancel PEA Session</button>
      )}

      {/* Error Display */}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* Standard Optimized Prompt Display */}
      {mode === 'standard' && optimizedPrompt && (
        <div>
          <h2>Optimized XML Prompt:</h2>
          <pre>{optimizedPrompt}</pre>
        </div>
      )}

      {/* PEA Chat Interface */}
      {mode === 'pea' && peaSessionId && peaInitialMessage && (
        <PEAChat
          sessionId={peaSessionId}
          initialMessage={peaInitialMessage}
          onFinalize={handleFinalizePrompt}
        />
      )}

      {/* Finalized XML Prompt Display (from PEA) */}
      {mode === 'pea' && finalXmlPrompt && (
        <div>
          <h2>Finalized XML Prompt from PEA:</h2>
          <pre>{finalXmlPrompt}</pre>
          {/* Optional: Button to copy or use the final prompt */}
        </div>
      )}
    </div>
  );
}

export default App;
