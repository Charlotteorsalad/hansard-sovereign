import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a minimal self-contained server bundle (.next/standalone) for Docker.
  output: "standalone",
};

export default nextConfig;
