"use client";

import dynamic from "next/dynamic";

const MapChart = dynamic(() => import("./MapChart"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[420px] bg-white border border-gray-200 rounded-lg flex items-center justify-center text-gray-400">
      <div className="flex flex-col items-center gap-2">
        <div className="w-6 h-6 border-2 border-gray-200 border-t-blue-500 rounded-full animate-spin"></div>
        <p className="text-sm">Loading map...</p>
      </div>
    </div>
  ),
});

export default function MapWrapper() {
  return <MapChart />;
}
