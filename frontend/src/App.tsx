import { useEffect, useMemo, useState } from 'react';
import { Routes, Route, useLocation, useNavigate, Link } from 'react-router-dom';

import { analyzePdf, askPhil } from './api';
import { UploadForm } from './components/UploadForm';
import { GraphView } from './components/GraphView';
import defaultGraph from './data/defaultGraph.json';
import type { GraphData, AskPhilResponse } from './types';

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
  const [answer, setAnswer] = useState<string | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);

  useEffect(() => {
    if (!question) {
      navigate('/', { replace: true });
      return;
    }

    let active = true;
    setLoading(true);
    setError(null);

    askPhil({ question, deepdag, graph: contextGraph })
      .then((response: AskPhilResponse) => {
        if (!active) return;
        setAnswer(response.answer);
        setGraph(response.subgraph);
      })
      .catch((err: unknown) => {
        if (!active) return;
        const message = err instanceof Error ? err.message : 'Unable to fetch explanation';
        setError(message);
      })
      .finally(() => {
        if (!active) return;
        setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [question, deepdag, navigate]);

  const heading = useMemo(() => (question ? `PhilDAG on: ${question}` : 'PhilDAG Assistant'), [question]);

  return (
    <main className="page">
      <Link to="/" className="assistant-back">← Back to workspace</Link>
      <h1>{heading}</h1>

      {loading && <p className="assistant-status">Thinking…</p>}
      {error && <p className="error">{error}</p>}
      {answer && <p className="assistant-answer">{answer}</p>}

      <GraphView graph={graph} />
    </main>
  );
}
