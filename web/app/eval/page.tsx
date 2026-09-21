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
  Square,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

type Hardware = { gpu: string; vram_mb: number; driver: string };
type ModelInfo = {
  name: string;
  size_mb: number;
  parameter_size: string;
  quantization: string;
};
type Info = {
  hardware: Hardware;
  models: ModelInfo[];
  queries: string[];
  production_model: string;
};

type Live = {
  ttft_ms: number | null;
  // load_ms/prefill_ms decompose ttft_ms (Ollama's own load_duration and
  // prompt_eval_duration) so a cold model load isn't mistaken for slow
  // prefill — on a cold run load_ms can be most of ttft_ms.
  load_ms: number | null;
  prefill_ms: number | null;
  tokens: number;
  tokens_per_sec: number;
  peak_vram_mb: number;
  processor: string;
  total_time_ms: number | null;
  done: boolean;
};

type Result = Live & { model: string; query: string };

// One point on the context-length curve (a single N).
type CtxPoint = {
  n_results: number;
  prompt_tokens: number;
  prefill_ms: number;
  total_time_ms: number;
  gen_tokens: number;
  peak_vram_mb: number;
  done: boolean;
};
const CTX_N = [3, 5, 10, 20, 30, 50];

const EMPTY: Live = {
  ttft_ms: null,
  load_ms: null,
  prefill_ms: null,
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
  signal?: AbortSignal,
) {
  const res = await fetch("/api/benchmark/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, query }),
    signal,
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
  prod,
}: {
  model: string;
  data: Live;
  maxTps: number;
  active: boolean;
  prod: boolean;
}) {
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
          <span>
            {/* Production keeps models resident, so a cold model load isn't
                what a real user experiences — lead with TTFT minus that load
                (what prefill alone costs), and note the load separately
                rather than silently folding it into the headline number. */}
            TTFT{" "}
            {fmt(
              Math.round(
                (data.ttft_ms ?? 0) - (data.load_ms && data.load_ms > 200 ? data.load_ms : 0)
              )
            )}{" "}
            ms
            {data.load_ms != null && data.load_ms > 200 && (
              <span className="text-amber-600">
                {" "}(+{fmt(Math.round(data.load_ms))} ms this run — cold model load)
              </span>
            )}
          </span>
          <span>total {fmt(Math.round(data.total_time_ms ?? 0))} ms</span>
          <span>VRAM {fmt(data.peak_vram_mb)} MB</span>
          <span className={onGpu(data.processor) ? "text-slate-500" : "text-red-500"}>
            {data.processor}
          </span>
        </div>
      )}
    </div>
  );
}

// One row of the context-length sweep: a prefill-latency bar for a given N.
function CtxRow({
  point,
  maxPrefill,
  active,
}: {
  point: CtxPoint;
  maxPrefill: number;
  active: boolean;
}) {
  const pct = maxPrefill ? Math.max((point.prefill_ms / maxPrefill) * 100, 2) : 2;
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-medium tabular-nums text-slate-900">
            N = {point.n_results}
          </span>
          {point.done && (
            <span className="text-xs text-slate-400">{fmt(point.prompt_tokens)} tok</span>
          )}
          {active && !point.done && (
            <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />
          )}
        </div>
        <div className="text-right">
          <span className="text-lg font-semibold tabular-nums text-slate-900">
            {point.done ? fmt(Math.round(point.prefill_ms)) : "—"}
          </span>
          <span className="ml-1 text-xs text-slate-400">ms prefill</span>
        </div>
      </div>
      <div className="mt-2 h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-[#c0504d] transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      {point.done && (
        <div className="mt-2 flex flex-wrap gap-x-4 text-xs text-slate-500">
          <span>total {fmt(Math.round(point.total_time_ms))} ms</span>
          <span>VRAM {fmt(point.peak_vram_mb)} MB</span>
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

  // Context-length sweep (second tab).
  const [tab, setTab] = useState<"models" | "context">("models");
  const [ctxModel, setCtxModel] = useState("");
  const [ctx, setCtx] = useState<Record<number, CtxPoint>>({});
  const [ctxCurrent, setCtxCurrent] = useState<number | null>(null);
  // True only during the pre-sweep warmup call, which can take several
  // seconds (a num_ctx switch forces a model reload) — ctxCurrent alone
  // can't show it, since it's only set once the real N points start.
  const [ctxWarming, setCtxWarming] = useState(false);

  const textBoxRef = useRef<HTMLDivElement>(null);
  // The in-flight request/loop, so a Stop click can cut in without clearing
  // whatever results (results/compare/ctx) are already on screen.
  const abortRef = useRef<AbortController | null>(null);

  function stopCurrent() {
    abortRef.current?.abort();
  }

  // The production RAG model — reported live by the backend (RAG_MODEL env
  // var), not hard-coded here, so this page can't drift out of sync with
  // whatever the backend is actually configured to serve.
  const isProd = (name: string) => name === info?.production_model;
  const prodModel = info?.models.find((m) => m.name === info.production_model);
  const vramGb =
    info && info.hardware.vram_mb > 0
      ? (info.hardware.vram_mb / 1024).toFixed(0)
      : null;

  useEffect(() => {
    fetch("/api/benchmark/info")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data: Info) => {
        setInfo(data);
        const def =
          data.models.find((m) => m.name === data.production_model)?.name ??
          data.models[0]?.name ??
          "";
        setModel(def);
        // The context sweep's whole point is isolating how prefill scales
        // with attention, not with CPU-offload noise — that only holds for
        // whichever model stays fully GPU-resident (see
        // docs/context_length_benchmark.md). The smallest installed model is
        // this project's stand-in for that; defaulting this tab to
        // production instead would drown the signal in CPU spill and make
        // long-N points slow without showing the effect they're there for.
        const smallest = [...data.models].sort((a, b) => a.size_mb - b.size_mb)[0]?.name ?? def;
        setCtxModel(smallest);
        setPresetQuery(data.queries[0] ?? "");
      })
      .catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    textBoxRef.current?.scrollTo({ top: textBoxRef.current.scrollHeight });
  }, [text]);

  const busy = running || compareCurrent !== null || ctxCurrent !== null || ctxWarming;

  async function run() {
    if (!model || !query || busy) return;
    const controller = new AbortController();
    abortRef.current = controller;
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
      }, controller.signal);
    } catch {
      // A Stop click aborts intentionally — that's not a failure to report.
      if (!controller.signal.aborted) {
        setError("Stream failed — is the backend still running?");
      }
    } finally {
      setRunning(false);
    }
  }

  async function compareAll() {
    if (!info || !query || busy) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setError("");
    // Seed every model with an empty row so the chart shows them all up front.
    const seeded: Record<string, Live> = {};
    for (const m of info.models) seeded[m.name] = { ...EMPTY };
    setCompare(seeded);

    for (const m of info.models) {
      if (controller.signal.aborted) break;
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
        }, controller.signal);
      } catch {
        // Stop was clicked — leave this model's partial row as-is, don't
        // report it as a failure, and don't start the next one.
        if (controller.signal.aborted) break;
        setError(`Run failed for ${m.name}.`);
      }
    }
    setCompareCurrent(null);
  }

  async function runContextSweep() {
    if (!ctxModel || !query || busy) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setError("");
    // Seed all N up front so the chart shows the axis before anything runs.
    const seeded: Record<number, CtxPoint> = {};
    for (const n of CTX_N) {
      seeded[n] = { n_results: n, prompt_tokens: 0, prefill_ms: 0,
        total_time_ms: 0, gen_tokens: 0, peak_vram_mb: 0, done: false };
    }
    setCtx(seeded);

    // Warm up before measuring: this sweep's CONTEXT_OPTIONS uses a different
    // num_ctx than the model-comparison tab (and than production), so if that
    // ran more recently, Ollama has to reload/reallocate for this num_ctx —
    // and the CPU-resident layers' weights get paged in from disk on first
    // touch. None of that is counted in load_duration, so it silently landed
    // inside N=3's prompt_eval_duration, making the very first point in every
    // sweep look far slower than it really is. One throwaway call absorbs
    // that cost before any point is actually recorded.
    setCtxWarming(true);
    try {
      await fetch("/api/benchmark/context", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: ctxModel, query, n_results: CTX_N[0] }),
        signal: controller.signal,
      });
    } catch {
      if (controller.signal.aborted) {
        setCtxWarming(false);
        setCtxCurrent(null);
        return;
      }
      // A failed warmup isn't fatal — the sweep below still runs, just with
      // N=3 possibly carrying the cold-start cost this was meant to absorb.
    } finally {
      setCtxWarming(false);
    }

    for (const n of CTX_N) {
      if (controller.signal.aborted) break;
      setCtxCurrent(n);
      try {
        const res = await fetch("/api/benchmark/context", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: ctxModel, query, n_results: n }),
          signal: controller.signal,
        });
        const d = (await res.json()) as CtxPoint;
        setCtx((prev) => ({ ...prev, [n]: { ...d, done: true } }));
      } catch {
        // Stop was clicked — leave N's row seeded-but-not-done, don't report
        // it as a failure, and don't start the next N.
        if (controller.signal.aborted) break;
        setError(`Context run failed at N=${n}.`);
      }
    }
    setCtxCurrent(null);
  }

  const compareModels = info?.models.map((m) => m.name) ?? [];
  const maxTps = Math.max(
    1,
    ...Object.values(compare).map((d) => d.tokens_per_sec),
  );
  const hasCompare = Object.keys(compare).length > 0;
  const hasCtx = Object.keys(ctx).length > 0;
  const maxPrefill = Math.max(1, ...Object.values(ctx).map((d) => d.prefill_ms));
  // Token/prefill growth ratio for the subtitle below — computed from this
  // sweep's own smallest/largest completed points, not a fixed figure from
  // one earlier run: it genuinely varies by model and query (a different
  // query's retrieved speeches aren't the same length), so a hard-coded
  // number would just be wrong most of the time.
  // Per-token cost (ms/token), not the raw endpoint ratio: with a fixed
  // decode-dominant FFN cost plus attention's O(n²) term only starting to
  // matter at the long end, "tokens rose 10× but prefill only rose 9×" reads
  // as sub-linear even though it isn't — per-token cost rising a modest ~15%
  // says the same thing (mild super-linear uptick) without that contradiction,
  // and without one noisy endpoint swinging a ratio of two raw totals.
  const ctxDone = Object.values(ctx).filter((p) => p.done);
  const ctxRatio =
    ctxDone.length >= 2
      ? (() => {
          const lo = ctxDone.reduce((a, b) => (a.n_results < b.n_results ? a : b));
          const hi = ctxDone.reduce((a, b) => (a.n_results > b.n_results ? a : b));
          if (lo.n_results === hi.n_results || lo.prompt_tokens === 0 || hi.prompt_tokens === 0) {
            return null;
          }
          const loMsPerTok = lo.prefill_ms / lo.prompt_tokens;
          const hiMsPerTok = hi.prefill_ms / hi.prompt_tokens;
          if (loMsPerTok <= 0) return null;
          return {
            loMsPerTok, hiMsPerTok,
            pct: ((hiMsPerTok - loMsPerTok) / loMsPerTok) * 100,
            loN: lo.n_results, hiN: hi.n_results,
          };
        })()
      : null;
  // The interpretation has to follow the sign of pct, not assume it: a model
  // spilling to CPU (pick one other than the fully GPU-resident one the
  // docs recommend for this sweep) adds its own per-token noise that can
  // flatten or even invert the attention-scaling signal this sweep is
  // meant to isolate — claiming "O(n²) is showing" when per-token cost
  // actually fell would contradict the very numbers next to it.
  const ctxTrend = !ctxRatio
    ? "attention's O(n²) term shows up as context grows"
    : ctxRatio.pct >= 10
      ? "mostly linear (the FFN term), with attention's O(n²) term visibly starting to show at the long end"
      : ctxRatio.pct >= -2
        ? "close to linear across this range — attention's O(n²) term isn't the dominant cost yet at these lengths"
        : "flat to slightly improving here — no super-linear signal in this run, likely CPU-offload noise masking it (the docs use a fully GPU-resident model to isolate the attention-scaling signal cleanly)";
  // Production model (llama) first in the dropdowns — it's the focus of the page.
  const orderedModels = info
    ? [...info.models].sort((a, b) => Number(isProd(b.name)) - Number(isProd(a.name)))
    : [];

  return (
    <div className="flex h-screen flex-col bg-slate-50">
      <header className="flex h-12 flex-shrink-0 items-center gap-3 border-b border-white/[0.06] bg-navy px-4">
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

      <div className="flex-1 overflow-y-auto">
      <main className="mx-auto max-w-4xl px-5 py-10">
        <div className="space-y-3">
          {info && (
            <Badge variant="secondary" className="gap-1.5">
              <Cpu className="h-3.5 w-3.5" />
              {info.hardware.gpu}
              {info.hardware.vram_mb > 0 && ` · ${fmt(info.hardware.vram_mb)} MB VRAM`}
            </Badge>
          )}
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">
            Live inference benchmark
          </h1>
          <p className="max-w-2xl text-slate-600">
            This RAG runs on{" "}
            <span className="font-medium text-slate-900">
              {info?.production_model ?? "…"}
            </span>
            {prodModel?.parameter_size && ` — ${prodModel.parameter_size} parameters`}
            {vramGb && `, on a ${vramGb} GB GPU`}
            . Pick a query and run it live, or compare every
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
                bash scripts/dev.sh
              </pre>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Tabs */}
            <div className="mt-8 flex w-fit gap-1 rounded-lg border border-slate-200 bg-white p-1">
              {(["models", "context"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  disabled={busy}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:opacity-50 ${
                    tab === t ? "bg-navy text-white" : "text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {t === "models" ? "Model comparison" : "Context length"}
                </button>
              ))}
            </div>

            {tab === "models" && (
            <>
            {/* Controls */}
            <Card className="mt-6 border-slate-200">
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
                      {orderedModels.map((m) => (
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
                  {(running || compareCurrent) && (
                    <Button
                      onClick={stopCurrent}
                      variant="destructive"
                      size="icon"
                      title="Stop — keeps whatever's already finished"
                    >
                      <Square className="h-3.5 w-3.5 fill-current" />
                    </Button>
                  )}
                </div>
                {compareCurrent && (
                  <p className="text-xs text-slate-500">
                    Models run one at a time —{" "}
                    {vramGb ? `only one fits in ${vramGb} GB at once` : "only one fits in VRAM at once"}
                    . This takes a couple of minutes
                    {prodModel?.parameter_size && `; the ${prodModel.parameter_size} spills to CPU and is slow`}.
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
                  Generation speed{vramGb && ` on a ${vramGb} GB GPU`}. Your
                  production model is{" "}
                  {prodModel?.parameter_size ? `the ${prodModel.parameter_size}` : "this one"}{" "}
                  — slower because it can&apos;t fully fit in VRAM, but chosen for answer
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
                      prod={isProd(m)}
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
                    // Minus any cold model-load time — production keeps
                    // models resident, so that's not what prefill costs.
                    value={
                      live.ttft_ms != null
                        ? fmt(
                            Math.round(
                              live.ttft_ms -
                                (live.load_ms && live.load_ms > 200 ? live.load_ms : 0)
                            )
                          )
                        : "—"
                    }
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
                {(live.processor || (live.load_ms != null && live.load_ms > 200)) && (
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    {live.processor && (
                      <Badge variant={onGpu(live.processor) ? "default" : "destructive"}>
                        layer split: {live.processor}
                      </Badge>
                    )}
                    {live.load_ms != null && live.load_ms > 200 && (
                      <span className="text-xs text-amber-600">
                        this run also spent {fmt(Math.round(live.load_ms))} ms loading the
                        model (cold) — already excluded from TTFT above; production keeps
                        models resident so this cost isn&apos;t paid per-request
                      </span>
                    )}
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
                        <th className="px-4 py-2.5 font-medium">total</th>
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
                            {/* Load-adjusted, same convention as the live tiles above —
                                a cold run's TTFT here would otherwise look erratic. */}
                            {fmt(
                              Math.round(
                                (r.ttft_ms ?? 0) - (r.load_ms && r.load_ms > 200 ? r.load_ms : 0)
                              )
                            )}{" "}
                            ms
                            {r.load_ms != null && r.load_ms > 200 && (
                              <span
                                className="text-amber-600"
                                title={`Cold run — ${fmt(Math.round(r.load_ms))} ms of model loading already excluded above`}
                              >
                                {" "}*
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-2.5 tabular-nums">
                            {fmt(Math.round(r.total_time_ms ?? 0))} ms
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

            {tab === "context" && (
              <>
                <Card className="mt-6 border-slate-200">
                  <CardContent className="space-y-4 py-5">
                    <div className="grid gap-4 sm:grid-cols-2">
                      <label className="flex flex-col gap-1.5 text-sm">
                        <span className="text-slate-500">
                          Model (defaults to smallest — stays fully in VRAM, for a clean signal)
                        </span>
                        <select
                          value={ctxModel}
                          onChange={(e) => setCtxModel(e.target.value)}
                          disabled={busy || !info}
                          className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-slate-900 outline-none focus:border-blue-500 disabled:opacity-50"
                        >
                          {orderedModels.map((m) => (
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
                    <div className="flex justify-center gap-2">
                      <Button onClick={runContextSweep} disabled={busy || !info} className="gap-1.5">
                        {ctxWarming || ctxCurrent ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Layers className="h-4 w-4" />
                        )}
                        {ctxWarming
                          ? "Warming up model…"
                          : ctxCurrent
                            ? `Running N=${ctxCurrent}…`
                            : "Run sweep (N = 3…50)"}
                      </Button>
                      {(ctxWarming || ctxCurrent !== null) && (
                        <Button
                          onClick={stopCurrent}
                          variant="destructive"
                          size="icon"
                          title="Stop — keeps whatever's already finished"
                        >
                          <Square className="h-3.5 w-3.5 fill-current" />
                        </Button>
                      )}
                    </div>
                    {ctxWarming && (
                      <p className="text-center text-xs text-amber-600">
                        Loading the model at this sweep&apos;s context size — a cold switch can
                        take several seconds. This isn&apos;t stuck.
                      </p>
                    )}
                    <p className="text-center text-xs text-slate-500">
                      Same query, more retrieved speeches → longer prompt. Measures how
                      prefill (TTFT) scales with context. The production{" "}
                      {prodModel?.parameter_size ?? "model"} spills to CPU, so its sweep is
                      slow — pick a smaller model above for a quick run.
                    </p>
                  </CardContent>
                </Card>

                {hasCtx && (
                  <section className="mt-6">
                    <h3 className="text-lg font-semibold text-slate-900">
                      Prefill latency vs context length
                    </h3>
                    <p className="mt-1 text-sm text-slate-500">
                      Bar length = prefill time.{" "}
                      {ctxRatio ? (
                        <>
                          Per-token prefill cost goes from {ctxRatio.loMsPerTok.toFixed(1)} to{" "}
                          {ctxRatio.hiMsPerTok.toFixed(1)} ms (N={ctxRatio.loN}→{ctxRatio.hiN},{" "}
                          {ctxRatio.pct >= 0 ? "+" : ""}
                          {ctxRatio.pct.toFixed(0)}%)
                        </>
                      ) : (
                        "Per-token prefill cost creeps up as context grows"
                      )}{" "}
                      — {ctxTrend}.
                    </p>
                    <div className="mt-3 space-y-2">
                      {CTX_N.map((n) => (
                        <CtxRow
                          key={n}
                          point={ctx[n] ?? { n_results: n, prompt_tokens: 0, prefill_ms: 0,
                            total_time_ms: 0, gen_tokens: 0, peak_vram_mb: 0, done: false }}
                          maxPrefill={maxPrefill}
                          active={ctxCurrent === n}
                        />
                      ))}
                    </div>
                    <p className="mt-4 text-xs text-slate-400">
                      Peak VRAM stays ~flat across N — Ollama pre-sizes the KV cache to
                      the context window, not the actual prompt. Full write-up in{" "}
                      <code>docs/context_length_benchmark.md</code>.
                    </p>
                  </section>
                )}
              </>
            )}

            {error && tab === "context" && (
              <p className="mt-3 text-sm text-red-600">{error}</p>
            )}
          </>
        )}
      </main>
      </div>
    </div>
  );
}
