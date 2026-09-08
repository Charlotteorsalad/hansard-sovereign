import { NextResponse } from "next/server";
import { BACKEND_URL } from "@/lib/backend";

// Never cache: the backend itself now rotates its picks every few days (not
// per request), so always ask it fresh rather than caching a stale answer.
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/suggestions`, {
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`backend responded ${res.status}`);
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    // Backend unreachable or still loading the model on startup; return no
    // suggestions instead of a 500.
    return NextResponse.json({ suggestions: [] });
  }
}
