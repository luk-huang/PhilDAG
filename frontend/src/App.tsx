import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Routes, Route, useLocation, useNavigate, Link } from 'react-router-dom';

import { analyzePdf, askPhil } from './api';
import { UploadForm } from './components/UploadForm';
import { GraphView } from './components/GraphView';
import defaultGraph from './data/defaultGraph.json';
import type { GraphData, ConversationTurn } from './types';
import { TTSButton } from './components/TTSButton';
import { useTTS } from './hooks/useTTS';

const SAMPLE_ANALYSIS = defaultGraph as GraphData;

export default function App() {
  const [graphData, setGraphData] = useState<GraphData | null>(SAMPLE_ANALYSIS);

  return (
    <Routes>
      <Route path="/" element={<GraphPage graphData={graphData} setGraphData={setGraphData} />} />
      <Route path="/assistant" element={<AssistantPage graphData={graphData} />} />
    </Routes>
  );
}

type GraphPageProps = {
  graphData: GraphData | null;
  setGraphData: (graph: GraphData | null) => void;
};

function GraphPage({ graphData, setGraphData }: GraphPageProps) {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [assistantPrompt, setAssistantPrompt] = useState('');
  const [deepDAGEnabled, setDeepDAGEnabled] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const navigate = useNavigate();

  const handleUpload = async (file: File) => {
    setError(null);
    setLoading(true);
    try {
      const result = await analyzePdf(file);
      setGraphData(result);
      setUploadedFiles((prev) => [...prev, file.name]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleAssistantSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const question = assistantPrompt.trim();
    if (!question) return;

    navigate('/assistant', {
      state: {
        question,
        deepdag: deepDAGEnabled,
        graph: graphData ?? SAMPLE_ANALYSIS,
      },
    });
    setAssistantPrompt('');
  };

  const handleResetGraph = () => {
    setGraphData(null);
    setUploadedFiles([]);
  };

  return (
    <>
      <main className="page">
        <h1>PhilDAG</h1>

        <UploadForm onSubmit={handleUpload} loading={loading} files={uploadedFiles} />

        {error && <p className="error">{error}</p>}

        <button className="graph-reset" type="button" onClick={handleResetGraph}>
          Reset graph
        </button>

        <GraphView graph={graphData} />
      </main>

      <div className="assistant-drawer">
        <form className="assistant-drawer__form" onSubmit={handleAssistantSubmit}>
          <input
            id="assistant-query"
            className="assistant-drawer__input"
            type="text"
            placeholder="Ask Phil"
            value={assistantPrompt}
            onChange={(event) => setAssistantPrompt(event.target.value)}
          />
          <button className="assistant-drawer__action" type="submit">
            Send
          </button>
        </form>

        <label className="assistant-drawer__toggle">
          <span className="assistant-drawer__toggle-label">DeepDAG</span>
          <input
            type="checkbox"
            checked={deepDAGEnabled}
            onChange={(event) => setDeepDAGEnabled(event.target.checked)}
          />
          <span className="assistant-drawer__toggle-track">
            <span className="assistant-drawer__toggle-thumb" />
          </span>
        </label>
      </div>
    </>
  );
}

type AssistantLocationState = {
  question: string;
  deepdag?: boolean;
  graph?: GraphData | null;
};

type AssistantPageProps = {
  graphData: GraphData | null;
};

function AssistantPage({ graphData }: AssistantPageProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as AssistantLocationState | null;
  const question = state?.question ?? '';
  const deepdag = state?.deepdag ?? false;
  const contextGraph = state?.graph ?? graphData ?? SAMPLE_ANALYSIS;

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [conversation, setConversation] = useState<ConversationTurn[]>([]);
  const [messageDraft, setMessageDraft] = useState('');
  const lastQuestionRef = useRef<string | null>(null);
  const tts = useTTS();
  const { speak: speakTTS, isLoading: ttsLoading, error: ttsError } = tts;
  const assistantSpokenCountRef = useRef<number>(0);

  const sendRequest = useCallback(
    async (prompt: string, history: ConversationTurn[]) => {
      setLoading(true);
      setError(null);
      try {
        const response = await askPhil({
          question: prompt,
          deepdag,
          graph: contextGraph,
          history,
        });
        setGraph(response.subgraph);
        setConversation((prev) => [...prev, { role: 'assistant', content: response.answer }]);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unable to fetch explanation';
        setError(message);
      } finally {
        setLoading(false);
      }
    },
    [deepdag, contextGraph],
  );

  useEffect(() => {
    const normalizedQuestion = question.trim();
    if (!normalizedQuestion) {
      navigate('/', { replace: true });
      return;
    }

    if (lastQuestionRef.current === normalizedQuestion) {
      return;
    }

    lastQuestionRef.current = normalizedQuestion;
    setConversation([{ role: 'user', content: normalizedQuestion }]);
    setMessageDraft('');
    setGraph(null);
    void sendRequest(normalizedQuestion, []);
  }, [question, navigate, sendRequest]);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const prompt = messageDraft.trim();
    if (!prompt) return;

    const history = conversation.slice();
    setConversation((prev) => [...prev, { role: 'user', content: prompt }]);
    setMessageDraft('');
    void sendRequest(prompt, history);
  };

  const heading = useMemo(() => {
    if (conversation.length > 0) {
      return `PhilDAG on: ${conversation[0].content}`;
    }
    return question ? `PhilDAG on: ${question}` : 'PhilDAG Assistant';
  }, [conversation, question]);

  const lastAssistantText = useMemo(() => {
    for (let i = conversation.length - 1; i >= 0; i -= 1) {
      if (conversation[i].role === 'assistant') {
        return conversation[i].content;
      }
    }
    return '';
  }, [conversation]);

  useEffect(() => {
    const assistantMessages = conversation.filter((m) => m.role === 'assistant');
    if (assistantMessages.length > assistantSpokenCountRef.current) {
      const latest = assistantMessages[assistantMessages.length - 1];
      if (latest?.content?.trim()) {
        void speakTTS(latest.content);
      }
      assistantSpokenCountRef.current = assistantMessages.length;
    }
  }, [conversation, speakTTS]);

  return (
    <main className="page">
      <Link to="/" className="assistant-back">← Back to workspace</Link>
      <h1>{heading}</h1>

      <section className="assistant-chat">
        {conversation.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`assistant-chat__message assistant-chat__message--${message.role}`}>
            <span>{message.content}</span>
            {message.role === 'assistant' && (
              <TTSButton text={message.content} controller={tts}>
                🔊 Speak
              </TTSButton>
            )}
          </div>
        ))}
        {loading && <p className="assistant-status">Thinking…</p>}
      </section>

      {error && <p className="error">{error}</p>}

      <form className="assistant-chat__form" onSubmit={handleSubmit}>
        <input
          className="assistant-chat__input"
          type="text"
          placeholder="Ask a follow-up"
          value={messageDraft}
          onChange={(event) => setMessageDraft(event.target.value)}
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          Send
        </button>
        <TTSButton
          text={lastAssistantText}
          controller={tts}
          disabled={ttsLoading || !lastAssistantText.trim()}
        >
          🔊 Read Last
        </TTSButton>
      </form>

      {ttsError && <p className="error">{ttsError}</p>}

      <GraphView graph={graph} />
    </main>
  );
}
