"use client";

import { useEffect, useState, use } from "react";
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
  date: string | number; // Allow number for timestamp transformation
  historical: number | null;
  forecast: number | null;
  lower: number | null;
  upper: number | null;
}

interface CountryDetails {
  country: string;
  id: string;
  points: DataPoint[];
}

export default function CountryPage({ params }: { params: Promise<{ id: string }> }) {
  // Unwrap params using `use()` hook as per Next.js 15/16 client component patterns
  const resolvedParams = use(params);
  const { id } = resolvedParams;
  
  const router = useRouter();
  const [data, setData] = useState<CountryDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch(`/data/details/${id}.json`)
      .then((res) => {
        if (!res.ok) throw new Error("Data not found");
        return res.json();
      })
      .then((data: CountryDetails) => {
        // Transform dates to timestamps for X-Axis scaling
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

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center text-white">
        <div className="w-8 h-8 border-4 border-slate-600 border-t-blue-500 rounded-full animate-spin"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center text-white gap-4">
        <h1 className="text-2xl font-bold text-red-400">Country Data Not Found</h1>
        <button 
          onClick={() => router.push("/")}
          className="px-4 py-2 bg-slate-800 rounded hover:bg-slate-700 transition-colors"
        >
          Back to Map
        </button>
      </div>
    );
  }

  const todayTimestamp = new Date().getTime();
  const lastDate = data.points.length > 0 ? (data.points[data.points.length - 1].date as number) : 0;
  const isDataStale = lastDate < todayTimestamp;

  return (
    <main className="min-h-screen bg-slate-950 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <button 
            onClick={() => router.push("/")}
            className="text-slate-400 hover:text-white mb-4 transition-colors flex items-center gap-2"
          >
            ← Back to Global Map
          </button>
          <div className="flex flex-col gap-2">
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
                {data.country} <span className="text-slate-600 text-2xl font-normal">({id})</span>
            </h1>
            {isDataStale && (
                <div className="inline-flex items-center gap-2 px-3 py-1 bg-amber-500/10 border border-amber-500/20 rounded-full text-amber-500 text-sm w-fit">
                    <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                    </span>
                    Dataset Outdated: Forecast ends in {format(new Date(lastDate), "MMMM yyyy")}
                </div>
            )}
          </div>
          <p className="text-slate-400 mt-2">
            Historical influenza cases and 12-month forecast
          </p>
        </header>

        <div className="w-full h-[500px] bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-xl">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={data.points}
              margin={{
                top: 20,
                right: 30,
                left: 20,
                bottom: 20,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis 
                dataKey="date" 
                type="number"
                domain={['dataMin', 'dataMax']}
                stroke="#94a3b8" 
                tickFormatter={(tick) => format(new Date(tick), "MMM yy")}
                minTickGap={50}
              />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155", color: "#f8fafc" }}
                labelFormatter={(label) => format(new Date(label), "MMM d, yyyy")}
                formatter={(value: number | undefined) => [value ? value.toFixed(0) : "0", "Cases"]}
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
                fill="#0f172a" 
                fillOpacity={1} 
              />

              {/* Historical Data */}
              <Line 
                type="monotone" 
                dataKey="historical" 
                stroke="#94a3b8" 
                strokeWidth={2} 
                dot={false} 
                name="Historical Cases" 
              />

              {/* Forecast Data */}
              <Line 
                type="monotone" 
                dataKey="forecast" 
                stroke="#3b82f6" 
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
