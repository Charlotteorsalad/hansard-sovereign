export async function GET() {
  try {
    const res = await fetch("http://localhost:8000/benchmark/info", {
      cache: "no-store",
      // Don't let a slow backend hang the page; fail to "offline" quickly.
      signal: AbortSignal.timeout(4000),
    });
    return new Response(res.body, {
      status: res.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    // Backend not running; let the page show an offline state.
    return Response.json({ error: "backend_offline" }, { status: 503 });
  }
}
