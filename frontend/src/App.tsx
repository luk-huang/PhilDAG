import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Routes, Route, useLocation, useNavigate, Link } from 'react-router-dom';

import { analyzePdf, askPhil } from './api';
import { UploadForm } from './components/UploadForm';
import { GraphView } from './components/GraphView';
import defaultGraph from './data/defaultGraph.json';
import type { GraphData, ConversationTurn } from './types';

const SAMPLE_ANALYSIS = defaultGraph as GraphData;

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<GraphPage />} />
      <Route path="/assistant" element={<AssistantPage />} />
    </Routes>
  );
}

function GraphPage() {
  const [graphData, setGraphData] = useState<GraphData | null>(SAMPLE_ANALYSIS);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [assistantPrompt, setAssistantPrompt] = useState('');
  const [deepDAGEnabled, setDeepDAGEnabled] = useState(false);
  const navigate = useNavigate();

  const handleUpload = async (file: File) => {
    setError(null);
    setLoading(true);
    try {
      const result = await analyzePdf(file);
      setGraphData(result);
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

  return (
    <>
      <main className="page">
        <h1>PhilDAG</h1>

        <UploadForm onSubmit={handleUpload} loading={loading} />

        {error && <p className="error">{error}</p>}

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

function AssistantPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as AssistantLocationState | null;
  const question = state?.question ?? '';
  const deepdag = state?.deepdag ?? false;
  const contextGraph = state?.graph ?? SAMPLE_ANALYSIS;

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [conversation, setConversation] = useState<ConversationTurn[]>([]);
  const [messageDraft, setMessageDraft] = useState('');
  const lastQuestionRef = useRef<string | null>(null);

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

  return (
    <main className="page">
      <Link to="/" className="assistant-back">← Back to workspace</Link>
      <h1>{heading}</h1>

      <section className="assistant-chat">
        {conversation.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`assistant-chat__message assistant-chat__message--${message.role}`}>
            <span>{message.content}</span>
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
      </form>

      <GraphView graph={graph} />
    </main>
  );
}
