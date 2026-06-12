// Thin client for the FastAPI backend.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Source {
  n: number;
  title: string;
  source: string;
  url?: string | null;
  snippet: string;
  dense_score?: number | null;
  sparse_score?: number | null;
}

export interface AgentStep {
  tool: string;
  arguments: Record<string, unknown>;
  summary: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  steps: AgentStep[];
  trace_id?: string | null;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export async function sendChat(
  message: string,
  history: ChatTurn[],
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export interface Health {
  status: string;
  documents_indexed: number;
  llm_configured: boolean;
  langfuse_enabled: boolean;
  model: string;
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

export interface ExperimentRow {
  model: string;
  kind: string;
  model_id?: string;
  chunk_size: number;
  chunk_overlap: number;
  collection_name?: string | null;
  persisted_chunks?: number;
  seconds: number;
  dim: number;
  num_chunks: number;
  dense_mrr: number;
  dense_hit: number;
  dense_recall?: number;
  dense_precision?: number;
  dense_ndcg?: number;
  hybrid_mrr: number;
  hybrid_hit: number;
  hybrid_recall?: number;
  hybrid_precision?: number;
  hybrid_ndcg?: number;
  per_category_hybrid_mrr: Record<string, number>;
}

export interface Experiments {
  status: string;
  generated_at?: string;
  k?: number;
  eval_size?: number;
  categories?: string[];
  corpus_sections?: number;
  distractors?: number;
  full_corpus?: boolean;
  persisted_vectors?: boolean;
  experiment_chroma_path?: string;
  total_configs: number;
  completed: number;
  skipped_models?: string[];
  results: ExperimentRow[];
}

export async function getExperiments(): Promise<Experiments> {
  const res = await fetch(`${API_URL}/experiments`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Experiments fetch failed (${res.status})`);
  return res.json();
}
