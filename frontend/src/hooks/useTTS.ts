import { useState, useCallback } from 'react';

interface TTSOptions {
  voiceId?: string;
  modelId?: string;
  voiceSettings?: {
    stability?: number;
    similarity_boost?: number;
  };
}

interface UseTTSReturn {
  speak: (text: string, options?: TTSOptions) => Promise<void>;
  isLoading: boolean;
  error: string | null;
  audioUrl: string | null;
  clearAudio: () => void;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const useTTS = (): UseTTSReturn => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const clearAudio = useCallback(() => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
  }, [audioUrl]);

  const speak = useCallback(async (text: string, options?: TTSOptions) => {
    if (!text.trim()) {
      setError('Please provide text to speak');
      return;
    }

    setIsLoading(true);
    setError(null);
    clearAudio();

    try {
      const response = await fetch(`${API_BASE_URL}/tts/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          voice_id: options?.voiceId,
          model_id: options?.modelId,
          voice_settings: options?.voiceSettings,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to generate speech' }));
        throw new Error(errorData.detail || 'Failed to generate speech');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

      // Auto-play the audio
      const audio = new Audio(url);
      audio.play().catch((err) => {
        console.error('Error playing audio:', err);
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      setError(errorMessage);
      console.error('TTS Error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [clearAudio]);

  return {
    speak,
    isLoading,
    error,
    audioUrl,
    clearAudio,
  };
};