import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the Turbopack workspace root to this directory. Without this, Next
  // detects the lockfile in the home directory (/home/david/package-lock.json)
  // and infers the wrong root, which breaks module resolution (e.g. tailwindcss)
  // and crashes the dev server on first page request.
  turbopack: {
    root: ".",
  },
};

export default nextConfig;
