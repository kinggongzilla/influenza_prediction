"use client";

import { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
  ReferenceLine,
  Label
} from "recharts";
import { format, parseISO } from "date-fns";

interface DataPoint {
  date: string | number;
  historical: number | null;
  forecast: number | null;
  lower: number | null;     // 90% prediction interval (10th percentile)
  upper: number | null;     // 90% prediction interval (90th percentile)
  lower_50: number | null;  // 50% prediction interval (25th percentile)
  upper_50: number | null;  // 50% prediction interval (75th percentile)
}

interface CountryData {
  country: string;
  id: string;
  data_type?: "ILI" | "ARI";
  stale?: boolean;
  last_update?: string | null;
  points: DataPoint[];
}

/** Human-readable data-freshness line for the page header,
    e.g. "Data last updated: week of 10 Aug 2026 (2 weeks ago)". */
function freshnessHint(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = parseISO(iso);
  const weeks = Math.round((Date.now() - d.getTime()) / (7 * 24 * 60 * 60 * 1000));
  const when = weeks <= 0 ? "this week" : weeks === 1 ? "1 week ago" : `${weeks} weeks ago`;
  return `Data last updated: week of ${format(d, "d MMM yyyy")} (${when})`;
}

/** Hover dot shared by all prediction-interval Areas. Recharts colors an
    Area's default activeDot with that Area's own fill, which would make the
    white "eraser" Areas (90%/50% lower lines) render blank white dots; the
    explicit dot keeps all four interval boundaries visually identical. */
const BAND_DOT = { r: 4, fill: "#3b82f6", stroke: "#ffffff", strokeWidth: 2 };

/** Custom tooltip: shows real prediction-interval *ranges* (lower–upper)
    and never the raw lower/upper dataKeys (those Areas exist only to
    draw the bands). Rows are skipped when their value is missing, so
    historical weeks show only the observed cases. */
function ChartTooltip(props: {
  active?: boolean;
  payload?: Array<{ payload?: DataPoint }>;
  label?: number;
  points: DataPoint[];
  dataType?: "ILI" | "ARI";
  showPI90: boolean;
  showPI50: boolean;
}) {
  const { active, payload, label, points, dataType, showPI90, showPI50 } = props;
  if (!active || !payload || payload.length === 0) return null;

  const entry = (payload[0]?.payload as DataPoint | undefined) ?? points.find((p) => p.date === label);
  if (!entry) return null;

  const num = (v: number | null | undefined): number | null =>
    typeof v === "number" && Number.isFinite(v) ? v : null;
  const fmt = (v: number) => Math.round(v).toLocaleString();

  const rows: { name: string; value: string; swatch: string }[] = [];
  const l90 = num(entry.lower), u90 = num(entry.upper);
  if (showPI90 && l90 != null && u90 != null)
    rows.push({ name: "90% prediction interval", value: `${fmt(l90)} – ${fmt(u90)}`, swatch: "#dbeafe" });
  const l50 = num(entry.lower_50), u50 = num(entry.upper_50);
  if (showPI50 && l50 != null && u50 != null)
    rows.push({ name: "50% prediction interval", value: `${fmt(l50)} – ${fmt(u50)}`, swatch: "#93c5fd" });
  const fc = num(entry.forecast);
  if (fc != null) rows.push({ name: "Forecast (mean)", value: fmt(fc), swatch: "#2563eb" });
  const hist = num(entry.historical);
  if (hist != null) rows.push({ name: dataType === "ARI" ? "Historical ARI cases" : "Historical ILI cases", value: fmt(hist), swatch: "#64748b" });

  if (rows.length === 0) return null;

  return (
    <div className="bg-white border border-gray-200 rounded-md shadow-md px-3 py-2 text-[13px] text-gray-800">
      <div className="font-semibold text-gray-500 mb-1">{format(new Date(label as number), "MMM d, yyyy")}</div>
      {rows.map((r) => (
        <div key={r.name} className="flex items-center gap-2 leading-5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm shrink-0" style={{ backgroundColor: r.swatch }} />
          <span className="text-gray-500">{r.name}:</span>
          <span className="font-medium">{r.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function CountryDetails({ id }: { id: string }) {

  const router = useRouter();
  const [data, setData] = useState<CountryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [timeRange, setTimeRange] = useState("3Y");
  const [showPI90, setShowPI90] = useState(true);
  const [showPI50, setShowPI50] = useState(true);

  useEffect(() => {
    fetch(`/data/details/${id}.json`)
      .then((res) => {
        if (!res.ok) throw new Error("Data not found");
        return res.json();
      })
      .then((data: CountryData) => {
        const transformedPoints = data.points.map(p => ({
            ...p,
            date: parseISO(p.date as string).getTime()
        }));
        setData({ ...data, points: transformedPoints });
        setLoading(false);
      })
      .catch(() => {
        setError(true);
        setLoading(false);
      });
  }, [id]);

  const filteredData = useMemo(() => {
    if (!data) return [];

    const points = data.points;
    if (points.length === 0) return [];

    // Each range = N weeks of the most recent HISTORY plus the near-term forecast
    // (8 weeks). Windows are anchored to the last historical week, not the end of
    // the series (which is the last forecast week). "All" shows everything,
    // including the full forecast horizon.
    const RANGES: Record<string, { hist: number; fc: number }> = {
      "1Y": { hist: 52, fc: 8 },
      "3Y": { hist: 156, fc: 8 },
      "5Y": { hist: 260, fc: 8 },
      "ALL": { hist: Infinity, fc: Infinity },
    };
    const t = RANGES[timeRange] ?? RANGES["1Y"];
    const hist = points.filter((p) => p.historical != null).slice(-t.hist);
    const fc = points.filter((p) => p.forecast != null).slice(0, t.fc);
    const rows = [...hist, ...fc].sort((a, b) => (a.date as number) - (b.date as number));

    // Bridge the forecast line to the last observed point: the forecast series
    // starts at the week after the last reported week, which on the time axis
    // leaves a one-week break between history and forecast. Copying the last
    // observed value onto its row makes the forecast start exactly where the
    // history ends (the prediction-interval band still begins at the first
    // genuine forecast week, one week ahead).
    const lastHist = [...rows].reverse().find((p) => p.historical != null);
    if (lastHist && rows.some((p) => p.date > lastHist.date && p.forecast != null)) {
      return rows.map((p) =>
        p.date === lastHist.date && p.historical != null && p.forecast == null
          ? { ...p, forecast: p.historical }
          : p
      );
    }
    return rows;
  }, [data, timeRange]);

  // Numeric y-domain spanning every visible series (history, forecast, both
  // PI bands). Feeds the Areas' baseValue: recharts' default 'auto' baseline
  // snaps to 0 whenever 0 is inside the domain, which erases the lower half
  // of any PI that straddles zero. Pinning all bands to the plot bottom keeps
  // every part of the interval visible (same range as the old 'auto' domain).
  const yDomain = useMemo(() => {
    const vals: number[] = [];
    for (const p of filteredData)
      for (const v of [p.historical, p.forecast, p.lower, p.lower_50, p.upper, p.upper_50])
        if (v != null && Number.isFinite(v)) vals.push(v);
    if (vals.length === 0) return [0, 1] as [number, number];
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (lo === hi) { lo -= 1; hi += 1; }
    return [lo, hi] as [number, number];
  }, [filteredData]);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#fafafa] flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-gray-200 border-t-blue-500 rounded-full animate-spin"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-[#fafafa] flex flex-col items-center justify-center gap-4">
        <h1 className="text-xl font-semibold text-red-600">Country Data Not Found</h1>
        <button
          onClick={() => router.push("/")}
          className="text-sm text-blue-600 hover:underline"
        >
          &larr; Back to Dashboard
        </button>
      </div>
    );
  }

  const todayTimestamp = new Date().getTime();
  const freshHint = freshnessHint(data.last_update);
  const windowStart = filteredData.length ? Math.min(...filteredData.map((p) => p.date as number)) : 0;
  const windowEnd = filteredData.length ? Math.max(...filteredData.map((p) => p.date as number)) : 0;
  const todayInView = todayTimestamp >= windowStart && todayTimestamp <= windowEnd;

  return (
    <main className="min-h-screen bg-[#fafafa]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <header className="mb-6">
          <button
            onClick={() => router.push("/")}
            className="text-sm text-blue-600 hover:underline mb-3 block"
          >
            &larr; Back to Dashboard
          </button>
          <div className="flex flex-col gap-3 md:flex-row md:justify-between md:items-end">
            <div>
                <h1 className="text-2xl font-semibold text-gray-900">
                    {data.country} <span className="text-gray-400 text-lg font-normal">({id})</span>
                </h1>
                <p className="text-gray-500 text-sm mt-1">
                    Historical {data.data_type === "ARI" ? "ARI (acute respiratory infection)" : "ILI (influenza-like illness)"} cases{data.stale ? " (no forecast; data is stale)" : " and 8-week forecast"}
                </p>
                {freshHint && (
                    <p className="text-xs text-gray-400 mt-1">{freshHint}</p>
                )}
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                <div className="flex items-center gap-2">
                <label htmlFor="timeRange" className="text-xs text-gray-500">Range:</label>
                <select
                    id="timeRange"
                    value={timeRange}
                    onChange={(e) => setTimeRange(e.target.value)}
                    className="bg-white border border-gray-300 text-gray-700 text-sm rounded-md px-2 py-1"
                >
                    <option value="1Y">1 Year</option>
                    <option value="3Y">3 Years</option>
                    <option value="5Y">5 Years</option>
                    <option value="ALL">All</option>
                </select>
                </div>
                <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-500">Prediction intervals:</span>
                    <label className="flex items-center gap-1.5 text-sm text-gray-700 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={showPI90}
                            onChange={(e) => setShowPI90(e.target.checked)}
                            className="accent-blue-600"
                        />
                        90%
                    </label>
                    <label className="flex items-center gap-1.5 text-sm text-gray-700 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={showPI50}
                            onChange={(e) => setShowPI50(e.target.checked)}
                            className="accent-blue-600"
                        />
                        50%
                    </label>
                </div>
            </div>
          </div>
        </header>

        {data.stale && (
          <div className="mb-4 bg-amber-50 border border-amber-300 text-amber-800 text-sm rounded-lg px-4 py-3">
            The latest surveillance data is dated <strong>{data.last_update ?? "unknown"}</strong> and is at least 8 weeks old.
            No forecast is shown because predictions are anchored to the latest data and only cover the next 8 weeks.
          </div>
        )}

        <div className="w-full h-[340px] sm:h-[500px] bg-white border border-gray-200 rounded-lg p-3 sm:p-4 shadow-sm">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={filteredData}
              margin={{
                top: 20,
                right: 30,
                left: 50,
                bottom: 45,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="date"
                type="number"
                domain={['dataMin', 'dataMax']}
                stroke="#94a3b8"
                tickFormatter={(tick) => format(new Date(tick), "MMM yy")}
                minTickGap={50}
                fontSize={12}
                label={{
                  value: "Date",
                  position: "insideBottom",
                  offset: 10,
                  style: { fill: "#94a3b8", fontSize: 12, textAnchor: "middle" },
                }}
              />
              <YAxis
                domain={yDomain}
                stroke="#94a3b8"
                fontSize={12}
                label={{
                  value: `Weekly ${data.data_type} cases`,
                  angle: -90,
                  position: "insideLeft",
                  offset: 8,
                  style: { fill: "#94a3b8", fontSize: 12, textAnchor: "middle" },
                }}
              />
              <Tooltip
                wrapperStyle={{ outline: "none" }}
                content={<ChartTooltip points={filteredData} dataType={data.data_type} showPI90={showPI90} showPI50={showPI50} />}
              />
              <Legend
                content={({ payload }) => {
                  const items = (payload ?? []).filter((item) =>
                    ["90% prediction interval", "50% prediction interval", "Forecast (Mean)", "Historical ILI Cases", "Historical ARI Cases"].includes(String(item.value))
                  );
                  return (
                    <ul className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-xs text-gray-500">
                      {items.map((item) => {
                        const v = String(item.value);
                        const isPI = v === "90% prediction interval" || v === "50% prediction interval";
                        return (
                          <li key={v} className="flex items-center gap-1.5">
                            <span
                              className="inline-block rounded-sm"
                              style={{
                                width: isPI ? 12 : 14,
                                height: isPI ? 12 : 3,
                                backgroundColor: v === "90% prediction interval"
                                  ? "#dbeafe"
                                  : v === "50% prediction interval"
                                    ? "#93c5fd"
                                    : v === "Forecast (Mean)"
                                      ? "#2563eb"
                                      : "#64748b",
                              }}
                            />
                            {v}
                          </li>
                        );
                      })}
                    </ul>
                  );
                }}
              />

              {/* Prediction intervals (toggle via checkboxes). A recharts Area
                  fills from its line down to its baseline, so each band is
                  built from a colored area plus an opaque "eraser" area
                  beneath it. All bands are pinned to the plot bottom with
                  baseValue (see yDomain). Opaque fills mean every region
                  between the four boundary lines carries exactly one color:
                  the 90% band pale blue, the 50% band darker blue, symmetric
                  above and below the mean. (With alpha fills the layers
                  stack and the lower shoulder always ends up darker.)
                  Trade-off: grid lines are hidden beneath the bands. */}
              {showPI90 && showPI50 && (
                <>
                  <Area type="monotone" dataKey="upper" stroke="none" fill="#dbeafe" fillOpacity={1} baseValue={yDomain[0]} activeDot={BAND_DOT} name="90% prediction interval" />
                  <Area type="monotone" dataKey="upper_50" stroke="none" fill="#93c5fd" fillOpacity={1} baseValue={yDomain[0]} activeDot={BAND_DOT} name="50% prediction interval" />
                  <Area type="monotone" dataKey="lower_50" stroke="none" fill="#dbeafe" fillOpacity={1} baseValue={yDomain[0]} activeDot={BAND_DOT} />
                  <Area type="monotone" dataKey="lower" stroke="none" fill="#ffffff" fillOpacity={1} baseValue={yDomain[0]} activeDot={BAND_DOT} />
                </>
              )}
              {showPI90 && !showPI50 && (
                <>
                  <Area type="monotone" dataKey="upper" stroke="none" fill="#dbeafe" fillOpacity={1} baseValue={yDomain[0]} activeDot={BAND_DOT} name="90% prediction interval" />
                  <Area type="monotone" dataKey="lower" stroke="none" fill="#ffffff" fillOpacity={1} baseValue={yDomain[0]} activeDot={BAND_DOT} />
                </>
              )}
              {showPI50 && !showPI90 && (
                <>
                  <Area type="monotone" dataKey="upper_50" stroke="none" fill="#93c5fd" fillOpacity={1} baseValue={yDomain[0]} activeDot={BAND_DOT} name="50% prediction interval" />
                  <Area type="monotone" dataKey="lower_50" stroke="none" fill="#ffffff" fillOpacity={1} baseValue={yDomain[0]} activeDot={BAND_DOT} />
                </>
              )}

              {/* Historical Data: dashed so it is distinguishable from the
                  forecast line by shape as well as color (redundant coding,
                  Color Universal Design) */}
              <Line
                type="monotone"
                dataKey="historical"
                stroke="#64748b"
                strokeWidth={2}
                strokeDasharray="6 4"
                dot={false}
                name={data.data_type === "ARI" ? "Historical ARI Cases" : "Historical ILI Cases"}
              />

              {/* Forecast Data */}
              <Line
                type="monotone"
                dataKey="forecast"
                stroke="#2563eb"
                strokeWidth={3}
                dot={false}
                name="Forecast (Mean)"
              />

              {/* Today Reference Line (only when inside the visible window). */}
              {/* Vermilion instead of red: red reads as dim/dark for red-green
                  colorblind users (Color Universal Design rule 4). */}
              {todayInView && (
                <ReferenceLine x={todayTimestamp} stroke="#D55E00" strokeDasharray="3 3">
                  <Label value="Today" position="insideTopLeft" fill="#D55E00" fontSize={12} />
                </ReferenceLine>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </main>
  );
}
