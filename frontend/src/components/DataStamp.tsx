"use client";

import { useEffect, useState } from "react";

type Status = {
  generated_at?: string;
  default_week?: string;
};

function fmt(d: string) {
  try {
    return new Date(d.replace(" ", "T")).toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  } catch {
    return d;
  }
}

export default function DataStamp() {
  const [status, setStatus] = useState<Status | null>(null);

  useEffect(() => {
    fetch("/data/influenza_status.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setStatus(d))
      .catch(() => {});
  }, []);

  if (!status?.generated_at) return null;

  return (
    <p className="text-xs text-gray-400">
      Data updated {fmt(status.generated_at)}
      {status.default_week ? ` · forecast week of ${fmt(status.default_week)}` : ""}
    </p>
  );
}
