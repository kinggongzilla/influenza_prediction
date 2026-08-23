import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin the Turbopack workspace root to this directory. Without this, Next
  // detects the lockfile in the home directory (/home/david/package-lock.json)
  // and infers the wrong root, which breaks module resolution (e.g. tailwindcss)
  // and crashes the dev server on first page request.
  turbopack: {
    root: "." ,
  },
  // Static export for the Cloudflare Pages deploy (set by auto_update.sh):
  //   STATIC_EXPORT=1 npx next build   ->   out/   ->   npx wrangler pages deploy out
  // Without the env var the build is the normal one used by `next start` locally.
  ...(process.env.STATIC_EXPORT === "1" ? { output: "export" as const } : {}),
};

export default nextConfig;
