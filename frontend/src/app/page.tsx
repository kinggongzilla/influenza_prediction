import MapWrapper from "@/components/MapWrapper";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-10 text-center">
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400 mb-2">
            Global Influenza Forecast
          </h1>
          <p className="text-slate-400">
            Real-time predicted influenza activity relative to historical trends.
          </p>
        </header>

        <section className="mb-12">
          <MapWrapper />
        </section>

        <footer className="text-center text-slate-500 text-sm mt-20">
          <p>Data source: WHO FluMart & Weather Data</p>
          <p>Powered by Mistral Vibe Test Pipeline</p>
        </footer>
      </div>
    </main>
  );
}