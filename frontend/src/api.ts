import type { GraphData, AskPhilRequest, AskPhilResponse } from './types';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export async function analyzePdf(file: File): Promise<GraphData> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/analyze/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json() as Promise<GraphData>;
}

export async function askPhil(req: AskPhilRequest): Promise<AskPhilResponse> {
  const response = await fetch(`${API_BASE}/ask/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json() as Promise<AskPhilResponse>;
}
