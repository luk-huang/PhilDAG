import { useState } from 'react';
import { analyzePdf } from './api';
import { UploadForm } from './components/UploadForm';
import { GraphView, type GraphData } from './components/GraphView';

const sampleArtifact = {
  id: 1,
  name: 'The Republic',
  author: 'Plato',
  tile: 'Allegory of the Cave',
  year: 'ca. 380 BCE',
};

const samplePremise1 = {
  id: 1,
  artifact: [sampleArtifact],
  statement: 'The prisoners only see shadows cast upon the cave wall, mistaking them for reality.',
  citations: [{ page: 1, text: 'Behold! human beings living in an underground den...' }],
};

const samplePremise2 = {
  id: 2,
  artifact: [sampleArtifact],
  statement: 'Leaving the cave is painful but reveals true forms under the sunlight.',
  citations: [{ page: 3, text: 'At first he will see the shadows best...' }],
};

const sampleConclusion = {
  id: 3,
  artifact: [sampleArtifact],
  statement: 'Education is the turning of the soul toward the Form of the Good.',
  citations: [{ page: 4, text: 'Education is not what some people boastfully assert it to be...' }],
};

const SAMPLE_ANALYSIS: GraphData = {
  statements: [samplePremise1, samplePremise2, sampleConclusion],
  arguments: [
    {
      id: 1,
      desc: 'Learning moves the soul from illusion to knowledge of the Good.',
      premise: [samplePremise1, samplePremise2],
      conclusion: sampleConclusion,
    },
  ],
};

export default function App() {
  const [graphData, setGraphData] = useState<GraphData | null>(SAMPLE_ANALYSIS);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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

  return (
    <main className="page">
      <h1>PhilDAG</h1>

      <UploadForm onSubmit={handleUpload} loading={loading} />

      {error && <p className="error">{error}</p>}

      <GraphView graph={graphData} />
    </main>
  );
}
