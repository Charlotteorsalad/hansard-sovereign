import { NextResponse } from "next/server";

export const dynamic = "force-dynamic"; // never cache; suggestions are randomised per request

export async function GET() {
  try {
    const res = await fetch("http://localhost:8000/suggestions", {
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
