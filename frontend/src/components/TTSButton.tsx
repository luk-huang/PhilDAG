import React from 'react';
import { useTTS, type UseTTSReturn } from '../hooks/useTTS';

interface TTSButtonProps {
  text: string;
  className?: string;
  disabled?: boolean;
  children?: React.ReactNode;
  controller?: UseTTSReturn;
}

export const TTSButton: React.FC<TTSButtonProps> = ({ 
  text, 
  className = '',
  disabled = false,
  children,
  controller,
}) => {
  const fallbackController = useTTS();
  const { speak, isLoading, isSpeaking, error } = controller ?? fallbackController;

  const handleClick = () => {
    speak(text);
  };

  return (
    <div>
      <button
        onClick={handleClick}
        disabled={disabled || isLoading || isSpeaking || !text}
        className={className}
      >
        {isLoading ? '🔊 Speaking...' : children || '🔊 Speak'}
      </button>
      {error && <p style={{ color: 'red', fontSize: '12px' }}>{error}</p>}
    </div>
  );
};
