"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  sendChat,
  getHealth,
  type ChatResponse,
  type ChatTurn,
  type Health,
} from "./lib/api";

interface Turn extends ChatTurn {
  sources?: ChatResponse["sources"];
  steps?: ChatResponse["steps"];
  traceId?: string | null;
}

const EXAMPLES = [
  "What are the funds-availability rules for next-day items under Regulation CC?",
  "When must a bank provide a Regulation Z adverse-action notice?",
  "What is the threshold for filing a Suspicious Activity Report?",
  "What disclosures does Regulation E require for electronic fund transfers?",
];

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading]);

  async function submit(text: string) {
    const message = text.trim();
    if (!message || loading) return;
    setError(null);
    setInput("");

    const history: ChatTurn[] = turns.map((t) => ({
      role: t.role,
      content: t.content,
    }));
    setTurns((prev) => [...prev, { role: "user", content: message }]);
    setLoading(true);

    try {
      const res = await sendChat(message, history);
      setTurns((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          steps: res.steps,
          traceId: res.trace_id,
        },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(input);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Agentic RAG · Banking Regulations Assistant</h1>
        <p>
          Hybrid search (vector + BM25) over banking regulations, with an agent
          that decides when to retrieve, calculate, or search the web.
        </p>
        <div className="nav">
          <Link href="/" className="active">
            Chat
          </Link>
          <Link href="/experiments">Benchmark</Link>
        </div>
        <div className="status">
          {health ? (
            <>
              <span className="pill">{health.documents_indexed} chunks indexed</span>
              <span className={`pill ${health.llm_configured ? "ok" : "warn"}`}>
                {health.llm_configured ? `LLM: ${health.model}` : "GROQ_API_KEY missing"}
              </span>
              <span className={`pill ${health.langfuse_enabled ? "ok" : ""}`}>
                Langfuse {health.langfuse_enabled ? "on" : "off"}
              </span>
            </>
          ) : (
            <span className="pill warn">backend offline</span>
          )}
        </div>
      </header>

      <div className="messages">
        {turns.length === 0 && (
          <div className="empty">
            Ask anything about U.S. banking regulations (Title 12 CFR — OCC,
            Federal Reserve, FDIC, NCUA, CFPB). The agent cites the regulation
            sections it used and shows its tool calls.
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button key={ex} onClick={() => submit(ex)}>
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((t, i) => (
          <div key={i} className={`msg ${t.role}`}>
            <div className="role">{t.role}</div>
            {t.steps && t.steps.length > 0 && (
              <div className="steps">
                {t.steps.map((s, j) => (
                  <span className="step" key={j}>
                    {s.tool} · {s.summary}
                  </span>
                ))}
              </div>
            )}
            <div className="bubble">{t.content}</div>
            {t.sources && t.sources.length > 0 && (
              <div className="sources">
                <div className="title">Sources</div>
                {t.sources.map((s) => (
                  <div className="source" key={s.n}>
                    <div className="src-head">
                      {s.url ? (
                        <a href={s.url} target="_blank" rel="noreferrer">
                          [{s.n}] {s.title}
                        </a>
                      ) : (
                        <span className="doc">
                          [{s.n}] {s.title}
                        </span>
                      )}
                      <span className="scores">
                        {s.dense_score != null && `dense ${s.dense_score}`}
                        {s.sparse_score != null && ` · bm25 ${s.sparse_score}`}
                      </span>
                    </div>
                    <div className="snippet">{s.snippet}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && <div className="typing">Thinking and retrieving…</div>}
        {error && <div className="error">{error}</div>}
        <div ref={endRef} />
      </div>

      <div className="composer">
        <textarea
          value={input}
          placeholder="Ask about the Claude API…  (Enter to send, Shift+Enter for newline)"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
        />
        <button onClick={() => submit(input)} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
