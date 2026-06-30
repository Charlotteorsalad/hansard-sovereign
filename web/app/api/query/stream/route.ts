import { NextRequest } from "next/server";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const res = await fetch("http://localhost:8000/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return new Response(res.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
