"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  getExperiments,
  type Experiments,
  type ExperimentRow,
} from "../lib/api";

type Metric = "hybrid_mrr" | "dense_mrr" | "hybrid_hit";
const METRICS: { key: Metric; label: string }[] = [
  { key: "hybrid_mrr", label: "Hybrid MRR" },
  { key: "dense_mrr", label: "Dense-only MRR" },
  { key: "hybrid_hit", label: "Hybrid Hit@5" },
];

type SortKey = keyof ExperimentRow;

function configLabel(r: ExperimentRow) {
  return `${r.model} · ${r.chunk_size}/${r.chunk_overlap}`;
}

export default function ExperimentsPage() {
  const [data, setData] = useState<Experiments | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<Metric>("hybrid_mrr");
  const [sortKey, setSortKey] = useState<SortKey>("hybrid_mrr");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    const load = async () => {
      try {
        const d = await getExperiments();
        setData(d);
        setError(null);
        if (d.status === "complete") clearInterval(timer);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load");
      }
    };
    load();
    timer = setInterval(load, 4000);
    return () => clearInterval(timer);
  }, []);

  const rows = data?.results ?? [];
  const bestMrr = useMemo(
    () => rows.reduce((m, r) => Math.max(m, r.hybrid_mrr), 0),
    [rows],
  );

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number")
        return (av - bv) * sortDir;
      return String(av).localeCompare(String(bv)) * sortDir;
    });
    return copy;
  }, [rows, sortKey, sortDir]);

  const barSorted = useMemo(
    () => [...rows].sort((a, b) => b[metric] - a[metric]),
    [rows, metric],
  );

  const models = useMemo(
    () => Array.from(new Set(rows.map((r) => r.model))),
    [rows],
  );
  const categories = data?.categories ?? [];
  // Heatmap: best per-category hybrid MRR for each model (max across chunkings).
  const heat = useMemo(() => {
    const m: Record<string, Record<string, number>> = {};
    for (const model of models) {
      m[model] = {};
      for (const cat of categories) {
        let best = 0;
        for (const r of rows)
          if (r.model === model)
            best = Math.max(best, r.per_category_hybrid_mrr?.[cat] ?? 0);
        m[model][cat] = best;
      }
    }
    return m;
  }, [rows, models, categories]);

  function header(key: SortKey, label: string) {
    const arrow = sortKey === key ? (sortDir === -1 ? " ▼" : " ▲") : "";
    return (
      <th
        onClick={() => {
          if (sortKey === key) setSortDir((d) => (d === 1 ? -1 : 1));
          else {
            setSortKey(key);
            setSortDir(-1);
          }
        }}
      >
        {label}
        {arrow}
      </th>
    );
  }

  const pct = data && data.total_configs
    ? Math.round((data.completed / data.total_configs) * 100)
    : 0;

  return (
    <div className="app wide">
      <header className="header">
        <h1>Embedding × Chunking Benchmark</h1>
        <p>
          How embedding model and chunk strategy affect retrieval on the 100-question
          banking-regulations eval. Dense-only isolates embedding quality; hybrid adds
          BM25 + RRF (what the app uses).
        </p>
        <div className="nav">
          <Link href="/">Chat</Link>
          <Link href="/experiments" className="active">
            Benchmark
          </Link>
        </div>
        <div className="notice">
          Hosted on a free backend, so the first request after idle can take
          30-40 seconds. Please give it a moment before clicking like the button
          owes you money.
        </div>
      </header>

      {!data && !error && <p className="typing">Loading results…</p>}
      {error && <div className="error">{error}</div>}

      {data && (
        <>
          <div className="summary">
            <span className="pill">{data.eval_size ?? 0} eval questions</span>
            <span className="pill">{data.corpus_sections ?? 0} corpus sections</span>
            <span className="pill">k = {data.k ?? 5}</span>
            <span className={`pill ${data.persisted_vectors ? "ok" : "warn"}`}>
              vectors {data.persisted_vectors ? "saved" : "temporary"}
            </span>
            {data.full_corpus && <span className="pill ok">full corpus</span>}
            <span className={`pill ${data.status === "complete" ? "ok" : "warn"}`}>
              {data.status} · {data.completed}/{data.total_configs} configs
            </span>
            {data.skipped_models && data.skipped_models.length > 0 && (
              <span className="pill warn">
                skipped (no key): {data.skipped_models.join(", ")}
              </span>
            )}
          </div>
          <div className="progress">
            <div style={{ width: `${pct}%` }} />
          </div>

          {rows.length === 0 ? (
            <p className="typing">
              No configs finished yet — results stream in as each completes.
            </p>
          ) : (
            <>
              <div className="controls">
                <span style={{ color: "var(--muted)" }}>Rank by:</span>
                {METRICS.map((m) => (
                  <button
                    key={m.key}
                    className={metric === m.key ? "on" : ""}
                    onClick={() => setMetric(m.key)}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              <div className="bars">
                {barSorted.map((r, i) => (
                  <div className="bar-row" key={i}>
                    <span className="bar-label">{configLabel(r)}</span>
                    <span className="bar-track">
                      <span
                        className="bar-fill"
                        style={{ width: `${Math.max(2, r[metric] * 100)}%` }}
                      />
                    </span>
                    <span className="bar-val">{r[metric].toFixed(3)}</span>
                  </div>
                ))}
              </div>

              <table className="grid">
                <thead>
                  <tr>
                    {header("model", "Model")}
                    {header("dim", "Dim")}
                    {header("chunk_size", "Chunk")}
                    {header("chunk_overlap", "Overlap")}
                    {header("num_chunks", "#Chunks")}
                    {header("seconds", "Embed s")}
                    {header("dense_mrr", "Dense MRR")}
                    {header("dense_hit", "Dense Hit")}
                    {header("dense_recall", "Dense Recall")}
                    {header("dense_precision", "Dense Precision")}
                    {header("dense_ndcg", "Dense NDCG")}
                    {header("hybrid_mrr", "Hybrid MRR")}
                    {header("hybrid_hit", "Hybrid Hit")}
                    {header("hybrid_recall", "Hybrid Recall")}
                    {header("hybrid_precision", "Hybrid Precision")}
                    {header("hybrid_ndcg", "Hybrid NDCG")}
                    <th>Collection</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((r, i) => (
                    <tr
                      key={i}
                      className={r.hybrid_mrr === bestMrr && bestMrr > 0 ? "best" : ""}
                    >
                      <td>
                        {r.model}
                        {r.kind === "openai" && <span className="tag-openai">API</span>}
                      </td>
                      <td>{r.dim}</td>
                      <td>{r.chunk_size}</td>
                      <td>{r.chunk_overlap}</td>
                      <td>{r.num_chunks.toLocaleString()}</td>
                      <td>{r.seconds}</td>
                      <td>{r.dense_mrr.toFixed(3)}</td>
                      <td>{r.dense_hit.toFixed(3)}</td>
                      <td>{(r.dense_recall ?? 0).toFixed(3)}</td>
                      <td>{(r.dense_precision ?? 0).toFixed(3)}</td>
                      <td>{(r.dense_ndcg ?? 0).toFixed(3)}</td>
                      <td>
                        <strong>{r.hybrid_mrr.toFixed(3)}</strong>
                      </td>
                      <td>{r.hybrid_hit.toFixed(3)}</td>
                      <td>{(r.hybrid_recall ?? 0).toFixed(3)}</td>
                      <td>{(r.hybrid_precision ?? 0).toFixed(3)}</td>
                      <td>{(r.hybrid_ndcg ?? 0).toFixed(3)}</td>
                      <td title={r.collection_name ?? ""}>
                        {r.collection_name ? (
                          <span className="collection-name">{r.collection_name}</span>
                        ) : (
                          "—"
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {categories.length > 0 && (
                <>
                  <h3 style={{ fontSize: 15, margin: "8px 0" }}>
                    Per-category hybrid MRR (best chunking per model)
                  </h3>
                  <table className="grid heat">
                    <thead>
                      <tr>
                        <th>Category</th>
                        {models.map((m) => (
                          <th key={m}>{m}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {categories.map((cat) => (
                        <tr key={cat}>
                          <td>{cat}</td>
                          {models.map((m) => {
                            const v = heat[m]?.[cat] ?? 0;
                            return (
                              <td
                                key={m}
                                style={{
                                  background: `rgba(217,119,87,${(v * 0.85).toFixed(2)})`,
                                  color: v > 0.5 ? "#1a1410" : "var(--text)",
                                }}
                              >
                                {v.toFixed(2)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
