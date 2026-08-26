export default function AboutPage() {
  return (
    <main className="min-h-screen bg-[#fafafa]">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-10">
        <h1 className="text-3xl font-semibold text-gray-900 mb-2">About</h1>
        <p className="text-gray-500 mb-10 max-w-2xl">
          An independent research project: one fine-tuned time-series foundation model
          producing probabilistic ILI and ARI forecasts for most of the world&apos;s countries
          without disease-specific design or local modelling expertise.
        </p>

        <section className="mb-10">
          <h2 className="text-xl font-semibold text-gray-900 mb-3">About the author</h2>
          <div className="flex items-start gap-4">
            <img
              src="/images/david_avatar.png"
              alt="David Hauser"
              className="h-20 w-20 rounded-full border border-gray-200 object-cover"
            />
            <p className="text-[15px] leading-7 text-gray-700">
              <strong>David Hauser</strong> is a machine learning researcher working at the
              intersection of AI and science, with a focus on time-series forecasting,
              physics simulation, and large language models.
              <span className="block mt-2">
                <a
                  href="https://github.com/kinggongzilla"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline mr-4"
                >
                  GitHub
                </a>
                <a
                  href="https://scholar.google.at/citations?user=pfICGIEAAAAJ&hl=en"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  Google Scholar
                </a>
              </span>
            </p>
          </div>
        </section>

        <section className="mb-10">
          <h2 className="text-xl font-semibold text-gray-900 mb-3">Conference abstract</h2>
          <div className="bg-white border border-gray-200 rounded-lg p-6 space-y-4">
            <p className="text-sm text-gray-500">
              An abstract describing this work was accepted for a poster presentation at{" "}
              <strong className="text-gray-700">ESCAIDE 2026</strong> (18&ndash;20 November, Stockholm).
            </p>
            <h3 className="text-lg font-semibold text-gray-900 leading-snug">
              A fine-tuned foundation model for influenza forecasting across 78 countries,
              benchmarked against Italy&apos;s collaborative forecasting hub
            </h3>
            <div className="space-y-3 text-[15px] leading-7 text-gray-700">
              <p>
                <strong>Background.</strong> Seasonal influenza causes substantial morbidity and
                mortality worldwide, and timely activity forecasts support preparedness and
                resource planning. Reliable forecasting models exist for only a few
                well-resourced countries, requiring disease-specific design and local expertise;
                most countries have no operational forecasts. We assessed whether a
                time-series foundation model could be fine-tuned into a single influenza
                forecaster for many countries and compared it with purpose-built national
                models.
              </p>
              <p>
                <strong>Methods.</strong> We fine-tuned Chronos-2, a pretrained time-series
                foundation model, on weekly influenza-like illness and acute respiratory
                infection records for 78 countries, selected from the WHO FluID global
                epidemiological surveillance database using predefined data-quality criteria.
                The model produced probabilistic four-week-ahead forecasts. We benchmarked it
                for the 2025/26 season against the models of Influcast, Italy&apos;s
                collaborative forecasting hub, using the pairwise relative weighted interval
                score (rWIS), where values below one beat the naive baseline. This season lay
                outside the fine-tuning data and post-dated Chronos-2&apos;s release, ensuring
                an out-of-sample test. We also evaluated Chronos-2 without fine-tuning
                (zero-shot).
              </p>
              <p>
                <strong>Results.</strong> Across 16 weekly forecasting rounds, the fine-tuned
                model ranked second of all entries, with a pairwise rWIS of 0.71, ahead of the
                Influcast ensemble (0.74) and eight of nine participating models, behind only
                one (0.70). The zero-shot model achieved an rWIS of 1.01, no better than the
                naive baseline; fine-tuning reduced the rWIS by 30% and mean absolute
                percentage error from 23% to 17%. Prediction intervals were over-confident:
                50% intervals covered 24% of observed values.
              </p>
              <p>
                <strong>Conclusions.</strong> A fine-tuned foundation model produced influenza
                forecasts that matched the purpose-built models of the Italian forecasting hub,
                whereas the off-the-shelf model performed no better than a naive baseline.
                Because one model forecasts across 78 countries, the approach could extend
                influenza forecasting to countries lacking dedicated modelling capacity,
                supporting preparedness and resource allocation. Benchmarking against
                forecasting models in other countries is needed to confirm whether this
                generalises.
              </p>
            </div>
          </div>
        </section>

        <section className="mb-10">
          <h2 className="text-xl font-semibold text-gray-900 mb-3">Data &amp; software</h2>
          <ul className="space-y-2 text-[15px] leading-7 text-gray-700">
            <li>
              <strong>Surveillance data:</strong> WHO FluID / FluMart, via the WHO Global
              Influenza Programme ({" "}
              <a
                href="https://www.who.int/teams/global-influenza-programme/surveillance-and-monitoring/influenza-surveillance-outputs"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                influenza-surveillance-outputs
              </a>
              ).
            </li>
            <li>
              <strong>Weather data (ablation studies only):</strong>{" "}
              <a
                href="https://open-meteo.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Open-Meteo
              </a>
              .
            </li>
            <li>
              <strong>Base model:</strong>{" "}
              <a
                href="https://github.com/amazon-science/chronos-forecasting"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Chronos-2
              </a>{" "}
              (Amazon Science).
            </li>
            <li>
              <strong>Comparison data:</strong> Influcast, the Italian epidemiological forecasting
              hub coordinated by the ISI Foundation in Turin, whose public forecasts we compare
              against.
            </li>
          </ul>
        </section>

        <section className="mb-10">
          <h2 className="text-xl font-semibold text-gray-900 mb-3">Citation</h2>
          <pre className="bg-white border border-gray-200 rounded-lg p-4 text-xs leading-5 text-gray-700 overflow-x-auto">
{`@misc{hauser2026flu,
      author = {Hauser, David},
      title  = {A Fine-tuned Foundation Model for Influenza Forecasting across 78
                 Countries, Benchmarked against Italy's Collaborative Forecasting Hub},
      year   = {2026},
      note   = {Abstract accepted for poster presentation at ESCAIDE 2026 (Stockholm, 18-20 Nov)}}`}
          </pre>
        </section>

        <section className="mb-10">
          <h2 className="text-xl font-semibold text-gray-900 mb-3">Contact</h2>
          <p className="text-[15px] leading-7 text-gray-700">
            Questions or corrections about the data or methodology? Reach out at{" "}
            <a href="mailto:hauser-david@hotmail.com" className="text-blue-600 hover:underline">
              hauser-david@hotmail.com
            </a>
            .
          </p>
        </section>
      </div>
    </main>
  );
}
