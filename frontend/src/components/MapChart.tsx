"use client";

import React, { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { ComposableMap, Geographies, Geography, ZoomableGroup, Sphere, Graticule } from "react-simple-maps";
import { interpolateRdYlGn } from "d3-scale-chromatic";
import { Tooltip } from "react-tooltip";

const geoUrl = "/data/countries-110m.json";

interface CountryData {
  id: string;
  numeric: string | null;
  name: string;
  value: number;
  threshold: number;
  score: number;
  status: "high" | "low";
}

const MapChart = () => {
  const router = useRouter();
  const [data, setData] = useState<CountryData[]>([]);
  const [tooltipContent, setTooltipContent] = useState("");
  const [position, setPosition] = useState({ coordinates: [0, 0] as [number, number], zoom: 1 });

  function handleZoomIn() {
    if (position.zoom >= 4) return;
    setPosition((pos) => ({ ...pos, zoom: pos.zoom * 1.2 }));
  }

  function handleZoomOut() {
    if (position.zoom <= 1) return;
    setPosition((pos) => ({ ...pos, zoom: pos.zoom / 1.2 }));
  }

  function handleMoveEnd(position: { coordinates: [number, number]; zoom: number }) {
    setPosition(position);
  }

  useEffect(() => {
    fetch("/data/influenza_status.json")
      .then((res) => res.json())
      .then((data) => {
        setData(data);
      })
      .catch(err => console.error("Failed to load data", err));
  }, []);

  // Create a map for quick lookup
  const dataMap = useMemo(() => {
    const map = new Map<number, CountryData>();
    data.forEach((d) => {
      if (d.numeric) {
        // Handle numeric codes: parse to int to handle "004" vs "4" mismatch
        map.set(parseInt(d.numeric, 10), d);
      }
    });
    return map;
  }, [data]);

  return (
    <div className="flex flex-col items-center">
      <div className="w-full max-w-5xl h-[600px] border border-gray-700 rounded-xl overflow-hidden bg-slate-900 shadow-2xl relative">
        <ComposableMap projectionConfig={{ rotate: [-10, 0, 0], scale: 150 }}>
            <ZoomableGroup
                zoom={position.zoom}
                center={position.coordinates}
                onMoveEnd={handleMoveEnd}
            >
                <Sphere stroke="#374151" strokeWidth={0.5} id="sphere" fill="transparent" />
                <Graticule stroke="#374151" strokeWidth={0.5} />
                <Geographies geography={geoUrl}>
                    {({ geographies }) =>
                    geographies.map((geo) => {
                        // geo.id is usually the numeric code in world-atlas
                        // If geo.id is missing or NaN, skip or use default
                        const geoId = geo.id ? parseInt(geo.id, 10) : -1;
                        const d = dataMap.get(geoId);
                        
                        let fillColor = "#374151"; // Default color (dark gray)
                        let hoverContent = "";

                        if (d) {
                            // interpolateRdYlGn: 0=Red, 1=Green
                            // score: 0=Low(Good), 1=High(Bad)
                            // We want 0->Green(1), 1->Red(0)
                            // So we use 1 - score
                            // Also clamping to avoid out of bounds just in case
                            const score = Math.max(0, Math.min(1, d.score));
                            fillColor = interpolateRdYlGn(1 - score);
                            
                            hoverContent = `${d.name}: Predicted ${d.value.toFixed(1)} (Score: ${(d.score * 100).toFixed(0)}%)`;
                        } else {
                            hoverContent = geo.properties.name || "Unknown Country";
                        }

                        return (
                        <Geography
                            key={geo.rsmKey}
                            geography={geo}
                            fill={fillColor}
                            stroke="#1F2937"
                            strokeWidth={0.5}
                            style={{
                                default: { outline: "none", transition: "all 250ms" },
                                hover: { fill: d ? fillColor : "#4B5563", outline: "none", filter: "brightness(1.2)", cursor: "pointer" },
                                pressed: { outline: "none" },
                            }}
                            onClick={() => {
                                if (d) {
                                    router.push(`/country/${d.id}`);
                                }
                            }}
                            onMouseEnter={() => {
                                setTooltipContent(hoverContent);
                            }}
                            onMouseLeave={() => {
                                setTooltipContent("");
                            }}
                            data-tooltip-id="map-tooltip"
                            data-tooltip-content={tooltipContent}
                        />
                        );
                    })
                    }
                </Geographies>
            </ZoomableGroup>
        </ComposableMap>
        
        {/* Zoom Controls */}
        <div className="absolute bottom-6 right-6 flex flex-col gap-2">
            <button
                onClick={handleZoomIn}
                className="bg-slate-800 hover:bg-slate-700 text-white p-2 rounded-lg shadow-lg border border-slate-600 transition-colors"
                aria-label="Zoom In"
            >
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
            </button>
            <button
                onClick={handleZoomOut}
                className="bg-slate-800 hover:bg-slate-700 text-white p-2 rounded-lg shadow-lg border border-slate-600 transition-colors"
                aria-label="Zoom Out"
            >
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
            </button>
        </div>
      </div>
      <Tooltip id="map-tooltip" style={{ backgroundColor: "#1F2937", color: "#F9FAFB", borderRadius: "8px" }} />
      
      <div className="mt-6 flex items-center gap-4 text-white">
        <span className="text-sm font-medium">Low Risk (Green)</span>
        <div className="w-64 h-4 rounded-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"></div>
        <span className="text-sm font-medium">High Risk (Red)</span>
      </div>
    </div>
  );
};

export default MapChart;
