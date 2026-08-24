"use client";

import React, { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { ComposableMap, Geographies, Geography, Graticule } from "react-simple-maps";
import { interpolateRdYlGn } from "d3-scale-chromatic";
import { Tooltip } from "react-tooltip";

const geoUrl = "/data/countries-110m.json";

interface ForecastWeek {
  date: string;
  value: number;
  zscore: number;
  score: number;
  status: "high" | "low";
}

interface CountryData {
  id: string;
  numeric: string | null;
  name: string;
  value: number | null;
  value_source?: "actual" | "forecast" | null;
  zscore: number | null;
  score: number | null;
  status: "high" | "low" | "stale";
  stale?: boolean;
  last_update?: string | null;
  data_type?: string;
  forecast_weeks?: ForecastWeek[];
}

interface MapData {
  generated_at?: string;
  default_week?: string | null;
  weeks?: string[];
  countries: CountryData[];
}

/** Format "2026-08-17" as "17 Aug 2026" without timezone pitfalls. */
const fmtWeek = (iso: string) => {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
};

const MapChart = () => {
  const router = useRouter();
  const [mapData, setMapData] = useState<MapData | null>(null);
  const [selectedWeek, setSelectedWeek] = useState<string | null>(null);
  const [tooltipContent, setTooltipContent] = useState("");

  useEffect(() => {
    fetch("/data/influenza_status.json")
      .then((res) => res.json())
      .then((data: MapData) => setMapData(data))
      .catch((err) => console.error("Failed to load data", err));
  }, []);

  const data = mapData?.countries ?? [];
  const weeks = mapData?.weeks ?? [];
  const defaultWeek = mapData?.default_week ?? null;
  // Future weeks no country can forecast (beyond every country's 4-week
  // horizon) would render an all-gray map, so they're hidden; they
  // reappear as the data gets fresher.
  const futureWeeks = weeks.filter(
    (w) => w !== defaultWeek && data.some((c) => c.forecast_weeks?.some((fw) => fw.date === w))
  );

  const dataMap = useMemo(() => {
    const map = new Map<number, CountryData>();
    data.forEach((d) => {
      const n = d.numeric != null ? parseInt(String(d.numeric), 10) : NaN;
      if (!Number.isNaN(n)) map.set(n, d);
    });
    return map;
  }, [data]);

  // The value shown for a country in the current view:
  // null view = "nowcast" (the calendar's current week: actual if
  // reported, otherwise the forecast for it); a selected week = that
  // country's forecast for exactly that week (future weeks are always
  // forecasts).
  const weekEntry = (d: CountryData): {
    value: number;
    zscore: number | null;
    score: number;
    status: string;
    date: string | null;
    source: "actual" | "forecast";
  } | null => {
    if (selectedWeek == null) {
      const v = d.value;
      const s = d.score;
      if (v == null || s == null) return null;
      return { value: v, zscore: d.zscore, score: s, status: d.status, date: null, source: d.value_source ?? "forecast" };
    }
    const w = d.forecast_weeks?.find((w) => w.date === selectedWeek) ?? null;
    if (!w) return null;
    return { value: w.value, zscore: w.zscore, score: w.score, status: w.status, date: w.date, source: "forecast" };
  };

  // Shared renderer for the world-map geographies. The UK is a single GB
  // polygon (its map value is the sum of the England/Scotland/N. Ireland
  // series — see generate_map_data.py).
  const renderGeography = (geo: any) => {
    const geoId = geo.id ? parseInt(geo.id, 10) : -1;
    const d = dataMap.get(geoId);
    const entry = d && !d.stale ? weekEntry(d) : null;

    let fillColor = "#e2e8f0";
    let hoverContent = "";

    if (d && !d.stale && entry) {
      const score = Math.max(0, Math.min(1, entry.score));
      fillColor = interpolateRdYlGn(1 - score);
      const dtype = d.data_type === "ARI" ? "ARI" : "ILI";
      const sign = entry.zscore != null && entry.zscore >= 0 ? "+" : "";
      const src = entry.source === "actual" ? "Actual" : "Predicted";
      if (selectedWeek) {
        hoverContent = `${d.name} [${dtype}]: ${src} ${entry.value.toFixed(1)} for week of ${fmtWeek(selectedWeek)} (${sign}${(entry.zscore ?? 0).toFixed(1)} SD vs historical mean)`;
      } else {
        hoverContent = `${d.name} [${dtype}]: Nowcast: ${src} ${entry.value.toFixed(1)} (${sign}${(entry.zscore ?? 0).toFixed(1)} SD)`;
      }
    } else if (d && d.stale) {
      const dtype = d.data_type === "ARI" ? "ARI" : "ILI";
      hoverContent = `${d.name} [${dtype}]: data last updated ${d.last_update ?? "unknown"} — no current forecast (data is 4+ weeks old)`;
    } else if (d && !d.stale && selectedWeek) {
      const dtype = d.data_type === "ARI" ? "ARI" : "ILI";
      hoverContent = `${d.name} [${dtype}]: no forecast for the week of ${fmtWeek(selectedWeek)}`;
    } else {
      hoverContent = geo.properties.name || "Unknown";
    }

    return (
      <Geography
        key={geo.rsmKey}
        geography={geo}
        fill={fillColor}
        stroke="#cbd5e1"
        strokeWidth={0.4}
        style={{
          default: { outline: "none", transition: "fill 200ms" },
          hover: {
            fill: d ? fillColor : "#cbd5e1",
            outline: "none",
            filter: d ? "brightness(0.9)" : "none",
            cursor: d ? "pointer" : "default",
          },
          pressed: { outline: "none" },
        }}
        onClick={() => {
          if (d) router.push(`/country/${d.id}`);
        }}
        onMouseEnter={() => setTooltipContent(hoverContent)}
        onMouseLeave={() => setTooltipContent("")}
        data-tooltip-id="map-tooltip"
        data-tooltip-content={tooltipContent}
      />
    );
  };

  return (
    <div className="flex flex-col items-center">
      <div className="w-full bg-white border border-gray-200 rounded-lg overflow-hidden shadow-sm">
        {/* Prediction-week selector */}
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-3 py-2.5 sm:px-4 sm:py-3 border-b border-gray-200 bg-gray-50">
          <div className="text-sm">
            <span className="font-medium text-gray-900">
              {selectedWeek
                ? `Predicted activity for the week of ${fmtWeek(selectedWeek)}`
                : `Nowcast — ${defaultWeek ? fmtWeek(defaultWeek) : "latest"} (actual where reported, otherwise predicted)`}
            </span>
          </div>
          <label className="flex items-center gap-2 text-sm text-gray-600">
            <span className="hidden sm:inline">Prediction week</span>
            <select
              value={selectedWeek ?? ""}
              onChange={(e) => setSelectedWeek(e.target.value || null)}
              className="border border-gray-300 rounded-md px-2 py-1 text-sm bg-white text-gray-900 focus:outline-none focus:ring-1 focus:ring-blue-400"
            >
              <option value="">Nowcast ({defaultWeek ? fmtWeek(defaultWeek) : "latest"})</option>
              {futureWeeks.map((w) => (
                <option key={w} value={w}>
                  Week of {fmtWeek(w)}
                </option>
              ))}
            </select>
          </label>
        </div>

        <ComposableMap
          projection="geoNaturalEarth1"
          // scale 140.6 = geoNaturalEarth1().fitExtent([[10,10],[790,410]], {type:"Sphere"})
          // for the 800x420 viewBox (RSM's default translate centers it):
          // whole globe incl. polar caps fits with a 10-unit margin.
          // The old hardcoded scale 160 overflowed the viewBox by ~18px
          // top/bottom and ~38px left/right, clipping the Arctic etc.
          projectionConfig={{ scale: 140.6 }}
          height={420}
        >
          <Graticule stroke="#e2e8f0" strokeWidth={0.4} />
          <Geographies geography={geoUrl}>
            {({ geographies }) => geographies.map(renderGeography)}
          </Geographies>
        </ComposableMap>
      </div>

      <Tooltip
        id="map-tooltip"
        style={{
          backgroundColor: "#ffffff",
          color: "#1a1a2e",
          borderRadius: "6px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
          fontSize: "13px",
          padding: "8px 12px",
        }}
      />

      {/* Legend */}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-x-3 gap-y-1.5 text-gray-600 text-sm">
        <span>Low</span>
        <div className="w-36 sm:w-48 h-3 rounded-full bg-gradient-to-r from-green-500 via-yellow-400 to-red-500 opacity-80"></div>
        <span>High</span>
        <span className="ml-4 sm:ml-6">{selectedWeek ? "No forecast for this week" : "No current data"}</span>
        <div className="w-6 h-3 rounded bg-slate-200 border border-slate-300"></div>
      </div>
    </div>
  );
};

export default MapChart;
