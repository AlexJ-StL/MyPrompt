import { useState } from 'react';
import './App.css';

/**
   * Renders the main application component that allows users to input a request,
   * optimizes the prompt via an API call, and displays the optimized result or an error.
   * 
   * @param {void} 
   * @returns {JSX.Element} The rendered application component.
   * @throws {Error} Throws an error if the API response is not ok.
   */function App() {
  const [userRequest, setUserRequest] = useState('');
  const [optimizedPrompt, setOptimizedPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleOptimizeClick = async () => {
    setLoading(true);
    setError(null);
    setOptimizedPrompt('');

    try {
      const response = await fetch('/api/optimize-prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ request: userRequest }),
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

  return (
    <div className="App">
      <h1>MyPrompt</h1>
      <textarea
        rows="10"
        cols="50"
        value={userRequest}
        onChange={(e) => setUserRequest(e.target.value)}
        placeholder="Enter your natural language request here..."
        disabled={loading}
      />
      <br />
      <button onClick={handleOptimizeClick} disabled={loading}>
        {loading ? 'Optimizing...' : 'Optimize Prompt'}
      </button>

      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {optimizedPrompt && (
        <div>
          <h2>Optimized XML Prompt:</h2>
          <pre>{optimizedPrompt}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
