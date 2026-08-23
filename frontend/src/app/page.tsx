import MapWrapper from "@/components/MapWrapper";
import DataStamp from "@/components/DataStamp";
import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#fafafa]">
      <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 sm:py-8">
        <section className="mb-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">
            Influenza Activity Dashboard
          </h1>
          <p className="text-gray-500 text-sm max-w-2xl">
            Current and near-term influenza activity for 78 countries, with quantified
            uncertainty. Compare countries week by week and look up to four weeks ahead;
            open any country for its detailed forecast and prediction intervals.{" "}
            <Link href="/methodology" className="text-blue-600 hover:underline">
              View methodology
            </Link>
          </p>
          <div className="mt-2">
            <DataStamp />
          </div>
        </section>

        <section className="mb-8">
          <MapWrapper />
        </section>

        <footer className="text-center text-gray-400 text-xs pt-8 border-t border-gray-200 space-y-1.5">
          <p className="flex items-center justify-center gap-1.5 flex-wrap">
            <span>Surveillance data:</span>
            <a
              href="https://www.who.int/teams/global-influenza-programme/surveillance-and-monitoring/influenza-surveillance-outputs"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-gray-600 hover:underline"
            >
              WHO FluID / FluMart
            </a>
            <span>· Weather data (research):</span>
            <a
              href="https://open-meteo.com"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-gray-600 hover:underline"
            >
              Open-Meteo
            </a>
            <span>· Model:</span>
            <a
              href="https://github.com/amazon-science/chronos-forecasting"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-gray-600 hover:underline"
            >
              Chronos-2
            </a>
          </p>
          <p>
            <Link href="/methodology" className="hover:text-gray-600 hover:underline">
              Methodology
            </Link>
            <span> · </span>
            <Link href="/about" className="hover:text-gray-600 hover:underline">
              About
            </Link>
          </p>
        </footer>
      </div>
    </main>
  );
}
