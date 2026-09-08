# Hansard Sovereign — web

Next.js (App Router) chat UI and `/eval` live inference benchmark for
[Hansard Sovereign](../README.md) — see the root README for what this project
is, architecture, and setup.

## Local development

```bash
npm install
npm run dev      # requires the FastAPI backend running on :8000 — see ../scripts/serve.sh
```

`BACKEND_URL` (default `http://localhost:8000`) points this app at the FastAPI
backend; see [lib/backend.ts](lib/backend.ts).

## Notes for this codebase specifically

See [AGENTS.md](AGENTS.md) — this repo runs a modified Next.js build with
breaking changes from the version most tooling assumes.
