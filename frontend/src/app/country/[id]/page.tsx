"use client";

import { useEffect, useState, use, useMemo } from "react";
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
import { format, parseISO, subYears } from "date-fns";

interface DataPoint {
  date: string | number;
  historical: number | null;
  forecast: number | null;
  lower: number | null;
  upper: number | null;
}

interface CountryDetails {
  country: string;
  id: string;
  data_type?: "ILI" | "ARI";
  points: DataPoint[];
}

export default function CountryPage({ params }: { params: Promise<{ id: string }> }) {
  const resolvedParams = use(params);
  const { id } = resolvedParams;

  const router = useRouter();
  const [data, setData] = useState<CountryDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [timeRange, setTimeRange] = useState("5Y");

  useEffect(() => {
    fetch(`/data/details/${id}.json`)
      .then((res) => {
        if (!res.ok) throw new Error("Data not found");
        return res.json();
      })
      .then((data: CountryDetails) => {
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
    if (timeRange === "ALL") return data.points;

    const points = data.points;
    if (points.length === 0) return [];

    const lastPoint = points[points.length - 1];
    const lastDate = new Date(lastPoint.date);
    let cutOffDate = 0;

    switch (timeRange) {
        case "1Y":
            cutOffDate = subYears(lastDate, 1).getTime();
            break;
        case "3Y":
            cutOffDate = subYears(lastDate, 3).getTime();
            break;
        case "5Y":
            cutOffDate = subYears(lastDate, 5).getTime();
            break;
        default:
            cutOffDate = 0;
    }

    return points.filter(p => (p.date as number) >= cutOffDate);
  }, [data, timeRange]);

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

  return (
    <main className="min-h-screen bg-[#fafafa]">
      <div className="max-w-7xl mx-auto px-6 py-6">
        <header className="mb-6">
          <button
            onClick={() => router.push("/")}
            className="text-sm text-blue-600 hover:underline mb-3 block"
          >
            &larr; Back to Dashboard
          </button>
          <div className="flex justify-between items-end">
            <div>
                <h1 className="text-2xl font-semibold text-gray-900">
                    {data.country} <span className="text-gray-400 text-lg font-normal">({id})</span>
                </h1>
                <p className="text-gray-500 text-sm mt-1">
                    Historical {data.data_type === "ARI" ? "ARI (acute respiratory infection)" : "ILI (influenza-like illness)"} cases and 12-month forecast
                </p>
            </div>
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
          </div>
        </header>

        <div className="w-full h-[500px] bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={filteredData}
              margin={{
                top: 20,
                right: 30,
                left: 20,
                bottom: 20,
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
              />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip
                contentStyle={{ backgroundColor: "#ffffff", borderColor: "#e2e8f0", color: "#1a1a2e", borderRadius: "6px", fontSize: "13px", boxShadow: "0 2px 8px rgba(0,0,0,0.08)" }}
                labelFormatter={(label) => format(new Date(label), "MMM d, yyyy")}
                formatter={(value: number | undefined) => [value ? value.toFixed(0) : "0", data.data_type === "ARI" ? "ARI Cases" : "ILI Cases"]}
              />
              <Legend />

              {/* Uncertainty Interval (10th-90th Percentile) */}
              <Area
                type="monotone"
                dataKey="upper"
                stroke="none"
                fill="#3b82f6"
                fillOpacity={0.1}
                name="Confidence Interval"
              />
              <Area
                type="monotone"
                dataKey="lower"
                stroke="none"
                fill="#ffffff"
                fillOpacity={1}
              />

              {/* Historical Data */}
              <Line
                type="monotone"
                dataKey="historical"
                stroke="#64748b"
                strokeWidth={2}
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

              {/* Today Reference Line */}
              <ReferenceLine x={todayTimestamp} stroke="#ef4444" strokeDasharray="3 3">
                <Label value="Today" position="insideTopLeft" fill="#ef4444" fontSize={12} />
              </ReferenceLine>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </main>
  );
}
