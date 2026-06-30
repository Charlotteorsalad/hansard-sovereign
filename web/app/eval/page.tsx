"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Cpu,
  Landmark,
  Play,
  Loader2,
  Trash2,
  Timer,
  Gauge,
  HardDrive,
  Hash,
  Star,
  Layers,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

type Hardware = { gpu: string; vram_mb: number; driver: string };
type ModelInfo = { name: string; size_mb: number };
type Info = { hardware: Hardware; models: ModelInfo[]; queries: string[] };

type Live = {
  ttft_ms: number | null;
  tokens: number;
  tokens_per_sec: number;
  peak_vram_mb: number;
  processor: string;
  total_time_ms: number | null;
  done: boolean;
};

type Result = Live & { model: string; query: string };

// The production RAG model, featured throughout so the page centres on it
// rather than on whatever happens to be fastest.
const PRODUCTION_MODEL = "llama3.1:8b-instruct-q4_K_M";
const isProd = (name: string) => name === PRODUCTION_MODEL;

const EMPTY: Live = {
  ttft_ms: null,
  tokens: 0,
  tokens_per_sec: 0,
  peak_vram_mb: 0,
  processor: "",
  total_time_ms: null,
  done: false,
};

const fmt = (n: number) => n.toLocaleString("en-US");
const onGpu = (p: string) => p.trim() === "100% GPU";

// Read an SSE stream from /benchmark/run, calling onEvent for each event.
async function streamRun(
  model: string,
  query: string,
  onEvent: (e: Record<string, unknown>) => void,
) {
  const res = await fetch("/api/benchmark/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, query }),
  });
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      onEvent(JSON.parse(line.slice(6)));
    }
  }
}

function StatTile({
  icon,
  label,
  value,
  unit,
  highlight,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  unit?: string;
  highlight?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
      <div className="flex items-center gap-1.5 text-xs text-slate-500">
        {icon}
        {label}
      </div>
      <div
        className={`mt-1 font-semibold tabular-nums ${
          highlight ? "text-3xl text-blue-600" : "text-2xl text-slate-900"
        }`}
      >
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-slate-400">{unit}</span>}
      </div>
    </div>
  );
}

// One row in the comparison chart: a tokens/sec bar with the model's metrics.
function CompareRow({
  model,
  data,
  maxTps,
  active,
}: {
  model: string;
  data: Live;
  maxTps: number;
  active: boolean;
}) {
  const prod = isProd(model);
  const color = prod ? "#2563eb" : data.done && !onGpu(data.processor) ? "#c0504d" : "#94a3b8";
  const pct = maxTps ? Math.max((data.tokens_per_sec / maxTps) * 100, 2) : 2;
  return (
    <div
      className={`rounded-xl border px-4 py-3 ${
        prod ? "border-blue-200 bg-blue-50/50" : "border-slate-200 bg-white"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="truncate font-medium text-slate-900">{model}</span>
          {prod && (
            <Badge className="shrink-0 gap-1">
              <Star className="h-3 w-3 fill-current" />
              production
            </Badge>
          )}
          {active && !data.done && (
            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-blue-500" />
          )}
        </div>
        <div className="shrink-0 text-right">
          <span className="text-lg font-semibold tabular-nums text-slate-900">
            {data.tokens_per_sec ? data.tokens_per_sec.toFixed(1) : "—"}
          </span>
          <span className="ml-1 text-xs text-slate-400">tok/s</span>
        </div>
      </div>
      <div className="mt-2 h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      {data.done && (
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-slate-500">
          <span>TTFT {fmt(Math.round(data.ttft_ms ?? 0))} ms</span>
          <span>VRAM {fmt(data.peak_vram_mb)} MB</span>
          <span className={onGpu(data.processor) ? "text-slate-500" : "text-red-500"}>
            {data.processor}
          </span>
        </div>
      )}
    </div>
  );
}

export default function EvalPage() {
  const [info, setInfo] = useState<Info | null>(null);
  const [offline, setOffline] = useState(false);
  const [model, setModel] = useState("");
  const [presetQuery, setPresetQuery] = useState(""); // curated dropdown pick
  const [customQuery, setCustomQuery] = useState(""); // free text, overrides preset
  const query = customQuery.trim() || presetQuery; // what actually gets run
  const [running, setRunning] = useState(false);
  const [live, setLive] = useState<Live>(EMPTY);
  const [text, setText] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [error, setError] = useState("");

  // Comparison mode: same query across every installed model, sequentially.
  const [compare, setCompare] = useState<Record<string, Live>>({});
  const [compareCurrent, setCompareCurrent] = useState<string | null>(null);
  const textBoxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/benchmark/info")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: Info) => {
        setInfo(data);
        const def =
          data.models.find((m) => isProd(m.name))?.name ??
          data.models[0]?.name ??
          "";
        setModel(def);
        setPresetQuery(data.queries[0] ?? "");
      })
      .catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    textBoxRef.current?.scrollTo({ top: textBoxRef.current.scrollHeight });
  }, [text]);

  const busy = running || compareCurrent !== null;

  async function run() {
    if (!model || !query || busy) return;
    setRunning(true);
    setLive(EMPTY);
    setText("");
    setError("");
    const acc: Live = { ...EMPTY };
    let body = "";
    try {
      await streamRun(model, query, (e) => {
        if (e.type === "first_token") acc.ttft_ms = e.ttft_ms as number;
        else if (e.type === "token") {
          acc.tokens = e.tokens as number;
          acc.tokens_per_sec = e.tokens_per_sec as number;
          acc.peak_vram_mb = e.peak_vram_mb as number;
          body += e.text as string;
          setText(body);
        } else if (e.type === "done") {
          Object.assign(acc, e, { done: true });
          setResults((prev) => [{ ...acc, model, query }, ...prev]);
        } else if (e.type === "error") setError(e.message as string);
        setLive({ ...acc });
      });
    } catch {
      setError("Stream failed — is the backend still running?");
    } finally {
      setRunning(false);
    }
  }

  async function compareAll() {
    if (!info || !query || busy) return;
    setError("");
    // Seed every model with an empty row so the chart shows them all up front.
    const seeded: Record<string, Live> = {};
    for (const m of info.models) seeded[m.name] = { ...EMPTY };
    setCompare(seeded);

    for (const m of info.models) {
      setCompareCurrent(m.name);
      const acc: Live = { ...EMPTY };
      try {
        await streamRun(m.name, query, (e) => {
          if (e.type === "first_token") acc.ttft_ms = e.ttft_ms as number;
          else if (e.type === "token") {
            acc.tokens = e.tokens as number;
            acc.tokens_per_sec = e.tokens_per_sec as number;
            acc.peak_vram_mb = e.peak_vram_mb as number;
          } else if (e.type === "done") {
            Object.assign(acc, e, { done: true });
          }
          setCompare((prev) => ({ ...prev, [m.name]: { ...acc } }));
        });
      } catch {
        setError(`Run failed for ${m.name}.`);
      }
    }
    setCompareCurrent(null);
  }

  const compareModels = info?.models.map((m) => m.name) ?? [];
  const maxTps = Math.max(
    1,
    ...Object.values(compare).map((d) => d.tokens_per_sec),
  );
  const hasCompare = Object.keys(compare).length > 0;

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="sticky top-0 z-10 flex h-12 items-center gap-3 border-b border-white/[0.06] bg-navy px-4">
        <Link
          href="/"
          className="flex items-center gap-1.5 text-sm text-slate-300 transition-colors hover:text-white"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to chat
        </Link>
        <Separator orientation="vertical" className="h-5 bg-white/10" />
        <div className="flex items-center gap-2">
          <Landmark className="h-4 w-4 text-blue-400" />
          <span className="text-sm font-semibold tracking-tight text-white">
            Hansard Sovereign
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-5 py-10">
        <div className="space-y-3">
          {info && (
            <Badge variant="secondary" className="gap-1.5">
              <Cpu className="h-3.5 w-3.5" />
              {info.hardware.gpu} · {fmt(info.hardware.vram_mb)} MB VRAM
            </Badge>
          )}
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">
            Live inference benchmark
          </h1>
          <p className="max-w-2xl text-slate-600">
            This RAG runs on{" "}
            <span className="font-medium text-slate-900">{PRODUCTION_MODEL}</span> — an
            8B model on a 4&nbsp;GB GPU. Pick a query and run it live, or compare every
            installed model on the same query. The numbers stream straight from the
            local LLM through the production retrieval path.
          </p>
        </div>

        {offline ? (
          <Card className="mt-8 border-amber-200 bg-amber-50">
            <CardContent className="space-y-2 py-5 text-sm text-amber-900">
              <p className="font-medium">Backend offline</p>
              <p>Start the FastAPI server so this page has something to benchmark:</p>
              <pre className="mt-1 overflow-x-auto rounded-lg bg-amber-900/90 px-3 py-2 font-mono text-xs text-amber-50">
                bash scripts/serve.sh
              </pre>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Controls */}
            <Card className="mt-8 border-slate-200">
              <CardContent className="space-y-4 py-5">
                <div className="grid gap-4 sm:grid-cols-2">
                  <label className="flex flex-col gap-1.5 text-sm">
                    <span className="text-slate-500">Model (single run)</span>
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      disabled={busy || !info}
                      className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-slate-900 outline-none focus:border-blue-500 disabled:opacity-50"
                    >
                      {info?.models.map((m) => (
                        <option key={m.name} value={m.name}>
                          {isProd(m.name) ? "★ " : ""}
                          {m.name} ({fmt(m.size_mb)} MB)
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="flex flex-col gap-1.5 text-sm">
                    <span className="text-slate-500">Query</span>
                    <select
                      value={presetQuery}
                      onChange={(e) => setPresetQuery(e.target.value)}
                      disabled={busy || !info}
                      className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-slate-900 outline-none focus:border-blue-500 disabled:opacity-50"
                    >
                      {info?.queries.map((q) => (
                        <option key={q} value={q}>
                          {q}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <input
                  value={customQuery}
                  onChange={(e) => setCustomQuery(e.target.value)}
                  disabled={busy || !info}
                  placeholder="…or type your own query (overrides the dropdown)"
                  className="h-9 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-900 outline-none focus:border-blue-500 disabled:opacity-50"
                />
                <div className="flex flex-wrap justify-center gap-2">
                  <Button onClick={run} disabled={busy || !info} className="gap-1.5">
                    {running ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                    {running ? "Running…" : "Run single"}
                  </Button>
                  <Button
                    onClick={compareAll}
                    disabled={busy || !info}
                    variant="outline"
                    className="gap-1.5"
                  >
                    {compareCurrent ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Layers className="h-4 w-4" />
                    )}
                    {compareCurrent
                      ? `Comparing ${compareCurrent}…`
                      : `Compare all ${info ? `(${info.models.length} models)` : ""}`}
                  </Button>
                </div>
                {compareCurrent && (
                  <p className="text-xs text-slate-500">
                    Models run one at a time — only one fits in 4&nbsp;GB at once. This
                    takes a couple of minutes; the 8B spills to CPU and is slow.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Comparison chart */}
            {hasCompare && (
              <section className="mt-6">
                <h3 className="text-lg font-semibold text-slate-900">
                  Same query, every model
                </h3>
                <p className="mt-1 text-sm text-slate-500">
                  Generation speed on a 4&nbsp;GB GPU. Your production model is the 8B —
                  slower because it can&apos;t fully fit in VRAM, but chosen for answer
                  quality over raw speed.
                </p>
                <div className="mt-3 space-y-2">
                  {compareModels.map((m) => (
                    <CompareRow
                      key={m}
                      model={m}
                      data={compare[m] ?? EMPTY}
                      maxTps={maxTps}
                      active={compareCurrent === m}
                    />
                  ))}
                </div>
              </section>
            )}

            {/* Single-run live tiles */}
            {(running || live.tokens > 0 || text) && (
              <>
                <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <StatTile
                    icon={<Gauge className="h-3.5 w-3.5" />}
                    label="tokens / sec"
                    value={live.tokens_per_sec ? live.tokens_per_sec.toFixed(1) : "—"}
                    highlight
                  />
                  <StatTile
                    icon={<Timer className="h-3.5 w-3.5" />}
                    label="time to first token"
                    value={live.ttft_ms != null ? fmt(Math.round(live.ttft_ms)) : "—"}
                    unit="ms"
                  />
                  <StatTile
                    icon={<Hash className="h-3.5 w-3.5" />}
                    label="tokens"
                    value={live.tokens ? fmt(live.tokens) : "—"}
                  />
                  <StatTile
                    icon={<HardDrive className="h-3.5 w-3.5" />}
                    label="peak VRAM"
                    value={live.peak_vram_mb ? fmt(live.peak_vram_mb) : "—"}
                    unit="MB"
                  />
                </div>
                {live.processor && (
                  <div className="mt-3">
                    <Badge variant={onGpu(live.processor) ? "default" : "destructive"}>
                      layer split: {live.processor}
                    </Badge>
                  </div>
                )}
                <div
                  ref={textBoxRef}
                  className="mt-4 max-h-48 overflow-y-auto rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm whitespace-pre-wrap text-slate-700"
                >
                  {text || <span className="text-slate-400">waiting for first token…</span>}
                </div>
              </>
            )}

            {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

            {/* Single-run history */}
            {results.length > 0 && (
              <section className="mt-10">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-slate-900">
                    Your single runs ({results.length})
                  </h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setResults([])}
                    className="gap-1.5 text-slate-500"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Clear
                  </Button>
                </div>
                <div className="mt-3 overflow-x-auto rounded-xl border border-slate-200 bg-white">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-left text-slate-500">
                        <th className="px-4 py-2.5 font-medium">Model</th>
                        <th className="px-4 py-2.5 font-medium">tok/s</th>
                        <th className="px-4 py-2.5 font-medium">TTFT</th>
                        <th className="px-4 py-2.5 font-medium">VRAM</th>
                        <th className="px-4 py-2.5 font-medium">Split</th>
                      </tr>
                    </thead>
                    <tbody className="text-slate-700">
                      {results.map((r, i) => (
                        <tr key={i} className="border-b border-slate-100 last:border-0">
                          <td className="px-4 py-2.5">
                            <span className="inline-flex items-center gap-1.5">
                              {isProd(r.model) && (
                                <Star className="h-3 w-3 shrink-0 fill-blue-500 text-blue-500" />
                              )}
                              <span
                                className="font-medium"
                                style={{ color: onGpu(r.processor) ? "#4f81bd" : "#c0504d" }}
                              >
                                {r.model}
                              </span>
                            </span>
                          </td>
                          <td className="px-4 py-2.5 font-semibold tabular-nums">
                            {r.tokens_per_sec.toFixed(1)}
                          </td>
                          <td className="px-4 py-2.5 tabular-nums">
                            {fmt(Math.round(r.ttft_ms ?? 0))} ms
                          </td>
                          <td className="px-4 py-2.5 tabular-nums">
                            {fmt(r.peak_vram_mb)} MB
                          </td>
                          <td className="px-4 py-2.5 text-xs">{r.processor}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            <p className="mt-10 text-xs text-slate-400">
              Single live runs vary with machine load. The averaged offline baseline
              (8 queries × 3 runs) lives in{" "}
              <code>results/quantization_benchmark.csv</code>; full write-up in{" "}
              <code>docs/quantization_benchmark.md</code>.
            </p>
          </>
        )}
      </main>
    </div>
  );
}
