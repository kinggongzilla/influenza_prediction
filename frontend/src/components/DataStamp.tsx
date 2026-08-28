"use client";

import { useEffect, useState } from "react";

type Status = {
  generated_at?: string;
  default_week?: string;
  latest_data_week?: string | null;
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

  // "Data updated <run date>" was misleading: the run happens daily even when
  // WHO has only added a week or two of reports. Stamp the newest week actually
  // reported to the feed instead (falls back to the run date if the field is
  // missing from an older JSON).
  const dataWeek = status.latest_data_week ?? status.generated_at;

  return (
    <p className="text-xs text-gray-400">
      Data through week of {fmt(dataWeek)} · varies by country
      {status.default_week ? ` · nowcast for week of ${fmt(status.default_week)}` : ""}
    </p>
  );
}
