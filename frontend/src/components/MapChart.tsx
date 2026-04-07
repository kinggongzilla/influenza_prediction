"use client";

import React, { useEffect, useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { ComposableMap, Geographies, Geography, Graticule } from "react-simple-maps";
import { interpolateRdYlGn } from "d3-scale-chromatic";
import { Tooltip } from "react-tooltip";

const geoUrl = "/data/countries-110m.json";

interface CountryData {
  id: string;
  numeric: string | null;
  name: string;
  value: number;
  zscore: number;
  score: number;
  status: "high" | "low";
  data_type?: string;
}

const MapChart = () => {
  const router = useRouter();
  const [data, setData] = useState<CountryData[]>([]);
  const [tooltipContent, setTooltipContent] = useState("");

  useEffect(() => {
    fetch("/data/influenza_status.json")
      .then((res) => res.json())
      .then((data) => setData(data))
      .catch((err) => console.error("Failed to load data", err));
  }, []);

  const dataMap = useMemo(() => {
    const map = new Map<number, CountryData>();
    data.forEach((d) => {
      if (d.numeric) {
        map.set(parseInt(d.numeric, 10), d);
      }
    });
    return map;
  }, [data]);

  return (
    <div className="flex flex-col items-center">
      <div className="w-full bg-white border border-gray-200 rounded-lg overflow-hidden shadow-sm">
        <ComposableMap
          projection="geoNaturalEarth1"
          projectionConfig={{ scale: 160 }}
          height={420}
        >
          <Graticule stroke="#e2e8f0" strokeWidth={0.4} />
          <Geographies geography={geoUrl}>
            {({ geographies }) =>
              geographies.map((geo) => {
                const geoId = geo.id ? parseInt(geo.id, 10) : -1;
                const d = dataMap.get(geoId);

                let fillColor = "#e2e8f0";
                let hoverContent = "";

                if (d) {
                  const score = Math.max(0, Math.min(1, d.score));
                  fillColor = interpolateRdYlGn(1 - score);
                  const dtype = d.data_type === "ARI" ? "ARI" : "ILI";
                  const sign = d.zscore >= 0 ? "+" : "";
                  hoverContent = `${d.name} [${dtype}]: Predicted ${d.value.toFixed(1)} (${sign}${d.zscore.toFixed(1)} SD)`;
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
              })
            }
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
      <div className="mt-4 flex items-center gap-3 text-gray-600 text-sm">
        <span>Low</span>
        <div className="w-48 h-3 rounded-full bg-gradient-to-r from-green-500 via-yellow-400 to-red-500 opacity-80"></div>
        <span>High</span>
      </div>
    </div>
  );
};

export default MapChart;
