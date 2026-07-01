// Base URL of the FastAPI backend. Defaults to localhost for `npm run dev`;
// set BACKEND_URL (e.g. http://backend:8000) when the frontend runs in Docker.
export const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";
