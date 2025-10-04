import React from 'react';
import { useTTS } from '../hooks/useTTS';

interface TTSButtonProps {
  text: string;
  className?: string;
  disabled?: boolean;
  children?: React.ReactNode;
}

export const TTSButton: React.FC<TTSButtonProps> = ({ 
  text, 
  className = '',
  disabled = false,
  children 
}) => {
  const { speak, isLoading, error } = useTTS();

  const handleClick = () => {
    speak(text);
  };

  return (
    <div>
      <button
        onClick={handleClick}
        disabled={disabled || isLoading || !text}
        className={className}
      >
        {isLoading ? '🔊 Speaking...' : children || '🔊 Speak'}
      </button>
      {error && <p style={{ color: 'red', fontSize: '12px' }}>{error}</p>}
    </div>
  );
};