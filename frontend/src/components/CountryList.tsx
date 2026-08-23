"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface Row {
  name: string;
  code: string;
  data_type: "ILI" | "ARI";
  since: string | null;
  last_update: string | null;
  stale: boolean;
  on_map: boolean;
}

interface CountryListData {
  generated_at: string;
  total: number;
  on_map: number;
  countries: Row[];
  excluded_from_training: string[];
}

export default function CountryList() {
  const [data, setData] = useState<CountryListData | null>(null);

  useEffect(() => {
    fetch("/data/country_list.json")
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => {});
  }, []);

  if (!data) {
    return <div className="my-4 text-sm text-gray-400">Loading country list…</div>;
  }

  return (
    <div>
      <div className="overflow-x-auto my-4">
        <table className="w-full text-sm border border-gray-200 rounded-lg overflow-hidden bg-white">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-3 py-2 font-medium text-left">Country</th>
              <th className="px-3 py-2 font-medium">Code</th>
              <th className="px-3 py-2 font-medium">Indicator</th>
              <th className="px-3 py-2 font-medium">Since</th>
              <th className="px-3 py-2 font-medium">Last update</th>
              <th className="px-3 py-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {data.countries.map((c) => (
              <tr key={c.code} className="border-b border-gray-100 last:border-0">
                <td className="px-3 py-1.5 whitespace-nowrap">
                  <Link href={`/country/${c.code}`} className="text-blue-600 hover:underline">
                    {c.name}
                  </Link>
                  {!c.on_map && (
                    <span className="ml-1.5 text-xs text-gray-400" title="No matching region on the 110m world map — linked here instead">
                      †
                    </span>
                  )}
                </td>
                <td className="px-3 py-1.5 text-gray-500">{c.code}</td>
                <td className="px-3 py-1.5">{c.data_type}</td>
                <td className="px-3 py-1.5 text-gray-500">{c.since ?? "—"}</td>
                <td className="px-3 py-1.5 text-gray-500">{c.last_update ?? "—"}</td>
                <td className="px-3 py-1.5">
                  {c.stale ? (
                    <span className="inline-block rounded-full bg-amber-100 text-amber-800 px-2 py-0.5 text-xs">
                      stale
                    </span>
                  ) : (
                    <span className="inline-block rounded-full bg-green-100 text-green-800 px-2 py-0.5 text-xs">
                      reporting
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-gray-400">
        † no matching region on the 110m world map (small-island territory or UK sub-region) —
        the forecast is available via the country page.
        {data.excluded_from_training.length > 0 && (
          <> Excluded from training after out-of-sample evaluation: {data.excluded_from_training.join(", ")}.</>
        )}
      </p>
    </div>
  );
}
