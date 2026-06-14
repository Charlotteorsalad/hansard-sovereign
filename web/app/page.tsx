"use client";

import { useState } from "react";

interface Source {
  speaker: string;
  date: string;
}

interface Result {
  answer: string;
  sources: Source[];
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleQuery() {
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);

    const res = await fetch("/api/query",{
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({query}), 
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  }

  return (
    <main className="max-w-3xl mx-auto p-8">
      <h1 className="text-2xl font-bold mb-2">Hansard Sovereign</h1>
      <p className="text-gray-500 mb-6">
        Ask questions about Malaysian Parliament debates
      </p>

      <div className="flex gap-2 mb-6">
        <input
          className="flex-1 border rounded px-4 py-2"
          placeholder="What did members say about public transport?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleQuery()}
        />
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded"
          onClick={handleQuery}
          disabled={loading}
        >
          {loading ? "..." : "Ask"}
        </button>
      </div>

      {result && (
        <div>
          <div className="bg-gray-50 rounded p-4 mb-4 whitespace-pre-wrap">
            {result.answer}
          </div>
          <div>
            <p className="text-sm font-semibold mb-2">Sources:</p>
            {result.sources.map((s, i) => (
              <div key={i} className="text-sm text-gray-600 mb-1">
                {s.speaker} — {s.date}
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}