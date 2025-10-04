import type { GraphData } from './components/GraphView';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export async function analyzePdf(file: File): Promise<GraphData> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const payload = (await response.json()) as GraphData;
  return payload;
}
