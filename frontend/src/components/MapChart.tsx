"use client";

import React, { useEffect, useState, useMemo, useRef } from "react";
import { useRouter } from "next/navigation";
import { ComposableMap, Geographies, Geography, Graticule } from "react-simple-maps";
import { interpolateRdYlBu } from "d3-scale-chromatic";
import { Tooltip } from "react-tooltip";

const geoUrl = "/data/countries-110m.json";

// Map viewBox size (must match the ComposableMap below) and zoom limits.
const MAP_W = 800;
const MAP_H = 420;
const MAX_ZOOM = 8;

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

  // ---- Mobile zoom/pan -------------------------------------------------
  // The world map has tiny tap targets for small-island countries, so on
  // touch devices the map is pinch-zoomable: one-finger swipes always scroll
  // the page (touch-action: pan-y) until the user zooms in, after which
  // one-finger moves pan the map (touch-action: none) and two fingers pinch.
  // Desktop is intentionally left static (no wheel/drag hijacking).
  const svgRef = useRef<SVGSVGElement | null>(null);
  const mapWrapRef = useRef<HTMLDivElement | null>(null);
  const [view, setView] = useState({ k: 1, x: 0, y: 0 });
  const [animating, setAnimating] = useState(false);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<{ k0: number; x0: number; y0: number; d0: number } | null>(null);
  const pan = useRef<{ x0: number; y0: number; sx: number; sy: number; px: number; py: number; moved: number } | null>(null);
  const suppressClick = useRef(false);

  // Clamp a transform so the (scaled) map always covers the viewBox.
  const clampView = (k: number, x: number, y: number) => {
    const kk = Math.min(MAX_ZOOM, Math.max(1, k));
    return {
      k: kk,
      x: Math.min(0, Math.max(MAP_W - MAP_W * kk, x)),
      y: Math.min(0, Math.max(MAP_H - MAP_H * kk, y)),
    };
  };

  // Client coordinates -> viewBox coordinates (robust to CSS scaling).
  const toSvg = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    return { x: p.x, y: p.y };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.pointerType === "mouse") return;
    const pt = toSvg(e);
    pointers.current.set(e.pointerId, pt);
    try {
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointers: ignore */
    }
    if (pointers.current.size === 2) {
      pan.current = null;
      setAnimating(false);
      const [a, b] = [...pointers.current.values()];
      pinch.current = { k0: view.k, x0: view.x, y0: view.y, d0: Math.max(Math.hypot(a.x - b.x, a.y - b.y), 1) };
    } else if (pointers.current.size === 1 && view.k > 1) {
      setAnimating(false);
      pan.current = { x0: view.x, y0: view.y, sx: pt.x, sy: pt.y, px: pt.x, py: pt.y, moved: 0 };
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (e.pointerType === "mouse" || !pointers.current.has(e.pointerId)) return;
    const pt = toSvg(e);
    pointers.current.set(e.pointerId, pt);
    if (pinch.current && pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()];
      const d = Math.max(Math.hypot(a.x - b.x, a.y - b.y), 1);
      const { k0, x0, y0, d0 } = pinch.current;
      const k = Math.min(MAX_ZOOM, Math.max(1, (k0 * d) / d0));
      // Keep the point under the pinch centroid stationary while zooming.
      const cx = (a.x + b.x) / 2;
      const cy = (a.y + b.y) / 2;
      setView(clampView(k, cx - (k / k0) * (cx - x0), cy - (k / k0) * (cy - y0)));
    } else if (pan.current) {
      const p = pan.current;
      p.moved += Math.abs(pt.x - p.px) + Math.abs(pt.y - p.py);
      p.px = pt.x;
      p.py = pt.y;
      // Absolute delta from the pan's start point (not incremental).
      setView(clampView(view.k, p.x0 + (p.px - p.sx), p.y0 + (p.py - p.sy)));
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (e.pointerType === "mouse") return;
    pointers.current.delete(e.pointerId);
    if (pinch.current && pointers.current.size < 2) {
      pinch.current = null;
      // Left one finger behind while zoomed in -> continue as a pan.
      if (pointers.current.size === 1 && view.k > 1) {
        const [a] = [...pointers.current.values()];
        pan.current = { x0: view.x, y0: view.y, sx: a.x, sy: a.y, px: a.x, py: a.y, moved: 0 };
      }
    }
    if (pan.current && pointers.current.size === 0) {
      // A real drag: swallow the click the browser fires afterwards, so a
      // pan that started on a country doesn't navigate to its page.
      if (pan.current.moved > 12) {
        suppressClick.current = true;
        window.setTimeout(() => (suppressClick.current = false), 400);
      }
      pan.current = null;
    }
  };

  const onClickCapture = (e: React.MouseEvent) => {
    if (suppressClick.current) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  // Zoom about the viewBox center (zoom controls).
  const zoomBy = (factor: number) => {
    const k1 = Math.min(MAX_ZOOM, Math.max(1, view.k * factor));
    if (k1 === view.k) return;
    const cx = MAP_W / 2;
    const cy = MAP_H / 2;
    setAnimating(true);
    setView(clampView(k1, cx - (k1 / view.k) * (cx - view.x), cy - (k1 / view.k) * (cy - view.y)));
  };

  const resetZoom = () => {
    setAnimating(true);
    setView({ k: 1, x: 0, y: 0 });
  };

  // RSM's ComposableMap doesn't forward a ref in its type, so grab the
  // rendered <svg> from its wrapper instead.
  useEffect(() => {
    svgRef.current = mapWrapRef.current?.querySelector("svg") ?? null;
  }, []);

  useEffect(() => {
    fetch("/data/influenza_status.json")
      .then((res) => res.json())
      .then((data: MapData) => setMapData(data))
      .catch((err) => console.error("Failed to load data", err));
  }, []);

  const data = mapData?.countries ?? [];
  const weeks = mapData?.weeks ?? [];
  const defaultWeek = mapData?.default_week ?? null;

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
  // series; see generate_map_data.py).
  const renderGeography = (geo: any) => {
    const geoId = geo.id ? parseInt(geo.id, 10) : -1;
    const d = dataMap.get(geoId);
    const entry = d && !d.stale ? weekEntry(d) : null;

    let fillColor = "#e2e8f0";
    let hoverContent = "";

    if (d && !d.stale && entry) {
      const score = Math.max(0, Math.min(1, entry.score));
      fillColor = interpolateRdYlBu(1 - score); // red (high) - pale (average) - blue (low): colorblind-safe (deuteranopia/protanopia simulation: ends stay separated)
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
      hoverContent = `${d.name} [${dtype}]: data last updated ${d.last_update ?? "unknown"}; no current forecast (data is 8+ weeks old)`;
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
        vectorEffect="non-scaling-stroke"
        className={d ? "active:brightness-90 [-webkit-tap-highlight-color:transparent]" : undefined}
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
                : `Nowcast: ${defaultWeek ? fmtWeek(defaultWeek) : "latest"} (actual where reported, otherwise predicted)`}
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
              {weeks.filter((w) => w !== defaultWeek).map((w) => (
                <option key={w} value={w}>
                  Week of {fmtWeek(w)}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="relative" ref={mapWrapRef}>
          <ComposableMap
            projection="geoNaturalEarth1"
            // scale 140.6 = geoNaturalEarth1().fitExtent([[10,10],[790,410]], {type:"Sphere"})
            // for the 800x420 viewBox (RSM's default translate centers it):
            // whole globe incl. polar caps fits with a 10-unit margin.
            // The old hardcoded scale 160 overflowed the viewBox by ~18px
            // top/bottom and ~38px left/right, clipping the Arctic etc.
            projectionConfig={{ scale: 140.6 }}
            height={420}
            style={{
              // pan-y: one-finger vertical swipes scroll the page, two-finger
              // pinch is always delivered; 'none' once zoomed in so one-finger
              // moves pan the map instead of the page.
              touchAction: view.k > 1 ? "none" : "pan-y",
              WebkitTouchCallout: "none",
            }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
            onClickCapture={onClickCapture}
          >
            <g
              style={{
                transform: `translate(${view.x}px, ${view.y}px) scale(${view.k})`,
                transformOrigin: "0px 0px",
                transition: animating ? "transform 250ms ease-out" : "none",
              }}
            >
              <Graticule stroke="#e2e8f0" strokeWidth={0.4} vectorEffect="non-scaling-stroke" />
              <Geographies geography={geoUrl}>
                {({ geographies }) => geographies.map(renderGeography)}
              </Geographies>
            </g>
          </ComposableMap>

          {/* Zoom controls: mobile only; desktop stays a static map. */}
          <div className="absolute right-2 top-2 flex flex-col gap-1.5 md:hidden">
            <button
              onClick={() => zoomBy(1.5)}
              disabled={view.k >= MAX_ZOOM}
              aria-label="Zoom in"
              className="h-11 w-11 rounded-lg border border-gray-300 bg-white/95 shadow-sm text-xl font-medium text-gray-700 active:bg-gray-100 disabled:opacity-40"
            >
              +
            </button>
            <button
              onClick={() => zoomBy(1 / 1.5)}
              disabled={view.k <= 1}
              aria-label="Zoom out"
              className="h-11 w-11 rounded-lg border border-gray-300 bg-white/95 shadow-sm text-xl font-medium text-gray-700 active:bg-gray-100 disabled:opacity-40"
            >
              −
            </button>
            <button
              onClick={resetZoom}
              disabled={view.k <= 1}
              aria-label="Reset map zoom"
              className="h-11 w-11 rounded-lg border border-gray-300 bg-white/95 shadow-sm text-lg text-gray-700 active:bg-gray-100 disabled:opacity-40"
            >
              ↺
            </button>
          </div>
          <div className="absolute bottom-2 left-2 rounded bg-white/90 px-1.5 py-0.5 text-[11px] text-gray-500 md:hidden">
            {view.k > 1 ? `zoomed ${view.k.toFixed(1)}× · drag to pan` : "pinch to zoom"}
          </div>
        </div>
      </div>

      <p className="mt-3 text-center text-xs text-gray-500 md:hidden">
        Tap a country to see its details
      </p>

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
        <div className="w-36 sm:w-48 h-3 rounded-full bg-gradient-to-r from-blue-700 via-yellow-100 to-red-700 opacity-80"></div>
        <span>High</span>
        <span className="ml-4 sm:ml-6">{selectedWeek ? "No forecast for this week" : "No current data"}</span>
        <div className="w-6 h-3 rounded bg-slate-200 border border-slate-300"></div>
      </div>
    </div>
  );
};

export default MapChart;
