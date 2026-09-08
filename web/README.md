# Hansard Sovereign — web

Next.js (App Router) chat UI and `/eval` live inference benchmark for
[Hansard Sovereign](../README.md) — see the root README for what this project
is, architecture, and full setup (including Docker).

## Local development

```bash
npm install
npm run dev      # requires the FastAPI backend running on :8000 — see ../scripts/serve.sh
```

Open <http://localhost:3000> to chat, or <http://localhost:3000/eval> for the
live inference benchmark. `BACKEND_URL` (default `http://localhost:8000`)
points this app at the FastAPI backend; see [lib/backend.ts](lib/backend.ts) —
set it to `http://host.docker.internal:8000` (or similar) when the frontend
runs somewhere the backend isn't on `localhost`.

## Pages

| Route | File | What it does |
| --- | --- | --- |
| `/` | [app/page.tsx](app/page.tsx) | The chat UI: streamed SSE answers with `[n]` citations, a model selector (8B vs the `hansard-qwen` fine-tune), **Compare mode** (both models answer the same question side by side), and a sidebar of past conversations. |
| `/eval` | [app/eval/page.tsx](app/eval/page.tsx) | Live inference benchmark: a **model comparison** tab (real generation against every installed Ollama model — TTFT, tokens/sec, peak VRAM) and a **context-length sweep** tab (prefill latency vs. number of retrieved speeches, N = 3…50). |

Both pages are client components; state (chat history, the active
view/compare state) is persisted to `localStorage` so a reload doesn't lose
the conversation.

## API routes

Everything under `app/api/` is a thin proxy to the FastAPI backend — it
exists so the browser only ever talks to same-origin Next.js, and so
`BACKEND_URL` can point at a different host without a CORS dance. SSE
endpoints pipe the backend's `text/event-stream` response straight through.

| Route | Proxies | Notes |
| --- | --- | --- |
| `GET /api/suggestions` | `GET /suggestions` | Example questions generated from real corpus facts; returns `{suggestions: []}` (not a 500) if the backend is still loading. |
| `POST /api/query` | `POST /query` | Non-streaming answer. |
| `POST /api/query/stream` | `POST /query/stream` | Streamed answer (SSE), used by both normal chat and Compare mode. |
| `GET /api/benchmark/info` | `GET /benchmark/info` | Hardware, installed models, benchmark query menu. Times out after 4s and returns `503` if the backend is offline, so the page can show an offline state instead of hanging. |
| `POST /api/benchmark/run` | `POST /benchmark/run` | Streamed live-benchmark run (SSE) for one model/query. |
| `POST /api/benchmark/context` | `POST /benchmark/context` | One point of the context-length sweep for a given `n_results`. |

## Tech stack

- **Next.js 16** (App Router, `output: "standalone"` for the Docker build) +
  **React 19**
- **Tailwind CSS 4** + [shadcn/ui](components.json) components in
  [components/ui/](components/ui/)
- No test runner is configured; `npm run lint` runs ESLint.

## Notes for this codebase specifically

See [AGENTS.md](AGENTS.md) — this repo runs a modified Next.js build with
breaking changes from the version most tooling assumes. Read the relevant
guide under `node_modules/next/dist/docs/` before writing Next.js-specific
code (routing, config, data fetching), since it may not match what you
already know about Next.js.
