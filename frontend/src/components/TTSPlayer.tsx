import React, { useState } from 'react';
import { useTTS } from '../hooks/useTTS';

export const TTSPlayer: React.FC = () => {
  const [text, setText] = useState('');
  const { speak, isLoading, error, audioUrl } = useTTS();

  const handleSpeak = () => {
    speak(text);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '600px' }}>
      <h2>Text to Speech (Alice)</h2>
      
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to convert to speech..."
        rows={6}
        style={{
          width: '100%',
          padding: '10px',
          borderRadius: '5px',
          border: '1px solid #ddd',
          fontSize: '14px',
          marginBottom: '10px',
        }}
      />

      <button
        onClick={handleSpeak}
        disabled={isLoading || !text.trim()}
        style={{
          padding: '12px 24px',
          backgroundColor: isLoading || !text.trim() ? '#ccc' : '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: isLoading || !text.trim() ? 'not-allowed' : 'pointer',
          fontSize: '16px',
        }}
      >
        {isLoading ? '🔊 Generating...' : '🔊 Generate Speech'}
      </button>

      {error && (
        <div style={{ color: '#d32f2f', marginTop: '10px' }}>
          Error: {error}
        </div>
      )}

      {audioUrl && (
        <audio
          controls
          src={audioUrl}
          style={{ width: '100%', marginTop: '20px' }}
        />
      )}
    </div>
  );
};