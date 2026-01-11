"use client";

import dynamic from "next/dynamic";

const MapChart = dynamic(() => import("./MapChart"), { 
  ssr: false,
  loading: () => (
    <div className="w-full max-w-5xl h-[600px] border border-gray-700 rounded-xl bg-slate-900 flex items-center justify-center text-slate-500">
      <div className="flex flex-col items-center gap-2">
        <div className="w-8 h-8 border-4 border-slate-600 border-t-blue-500 rounded-full animate-spin"></div>
        <p>Loading Interactive Map...</p>
      </div>
    </div>
  )
});

export default function MapWrapper() {
  return <MapChart />;
}
