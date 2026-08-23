import type { ReactNode } from "react";
import CountryList from "@/components/CountryList";

const models = [
  { name: "ISI-FluBcast", rwis: 0.7, note: "", rank: 1 },
  { name: "Chronos-2 fine-tuned (this work)", rwis: 0.71, note: "us", rank: 2 },
  { name: "Influcast ensemble", rwis: 0.74, note: "ensemble", rank: 3 },
  { name: "Chronos-2 fine-tuned (no week-of-year covariate)", rwis: 0.75, note: "", rank: 4 },
  { name: "ISI-FluABCaster", rwis: 0.91, note: "", rank: 5 },
  { name: "Influcast quantile baseline", rwis: 1.0, note: "baseline", rank: 6 },
  { name: "C2S2_Trento-SIR_INN", rwis: 1.0, note: "", rank: 7 },
  { name: "comunipd-mobnetSI2R", rwis: 1.07, note: "", rank: 8 },
  { name: "CSL_PoliTo-metaFlu", rwis: 1.11, note: "", rank: 9 },
  { name: "ISI-GLEAM", rwis: 1.22, note: "", rank: 10 },
  { name: "EpiQMUL-SEIR_QMUL", rwis: 1.28, note: "", rank: 11 },
  { name: "ISI-IPSICast", rwis: 1.32, note: "", rank: 12 },
  { name: "UNIPD_NEIDE-SEEIIRS_MCMC", rwis: 1.45, note: "", rank: 13 },
];

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="text-xl font-semibold text-gray-900 mb-3">{title}</h2>
      <div className="space-y-3 text-[15px] leading-7 text-gray-700">{children}</div>
    </section>
  );
}

export default function MethodologyPage() {
  return (
    <main className="min-h-screen bg-[#fafafa]">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-10">
        <h1 className="text-3xl font-semibold text-gray-900 mb-2">Methodology</h1>
        <p className="text-gray-500 mb-10 max-w-2xl">
          How the forecasts on this dashboard are built, what the numbers mean, and where the
          approach breaks down. Every figure below is reproducible from the project data and
          evaluation pipeline.
        </p>

        <Section title="1. Surveillance data">
          <p>
            The model is trained on <strong>weekly influenza-like illness (ILI) and acute
            respiratory infection (ARI) case counts</strong> from the WHO&apos;s{" "}
            <a
              href="https://www.who.int/teams/global-influenza-programme/surveillance-and-monitoring/influenza-surveillance-outputs"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              FluID / FluMart global surveillance database
            </a>
            . Countries report either ILI (a clinical syndrome) or ARI (a broader respiratory
            category); countries that switched indicator mid-series (e.g. Italy, ILI to ARI in
            late 2025) are kept as a single continuous series with a marker for which indicator
            each week belongs to, so the model can treat the regimes differently.
          </p>
          <p>
            Not every country&apos;s series is usable. Candidates were screened with fixed,
            predefined quality criteria, and <strong>89 countries passed</strong>:
          </p>
          <ul className="list-disc pl-6 space-y-1.5">
            <li>at least 156 non-missing weeks (≈ 3 years of actual data)</li>
            <li>spanning at least 4 years of coverage</li>
            <li>at most 50% of weeks missing for irregularly reporting series</li>
            <li>median non-zero weekly count of at least 20 (excludes micro-series with near-zero counts)</li>
          </ul>
          <p>
            Selection criteria are applied identically to all countries and are deliberately
            data-driven rather than judgment-based, to avoid cherry-picking easy-to-predict
            series. 72 of the 89 can be color-coded on the world map; the remaining 17
            (small-island territories and the UK sub-regions that WHO reports separately)
            have no matching region in the 110m map geography, so their forecasts are linked
            from the list below instead.
          </p>
          <CountryList />
        </Section>

        <Section title="2. Model">
          <p>
            The forecaster is{" "}
            <a
              href="https://github.com/amazon-science/chronos-forecasting"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              Chronos-2
            </a>
            , a pretrained time-series foundation model (patch-based transformer trained with a
            quantile-regression loss, so its outputs are probabilistic by construction). It is
            fine-tuned on the 89-country series set; the production configuration conditions on
            two covariates, the indicator type (ILI vs ARI) and hemisphere, selected by
            ablation. Additional covariates (week-of-year, historical weather from Open-Meteo,
            neighbouring-country activity) were evaluated and did not improve
            out-of-sample scores, so they are not used.
          </p>
          <p>
            The design goal is <em>one model, many countries</em>: no disease-specific model, no
            local calibration loop, and no country-specific hyperparameters. Countries with
            sparse or low-quality series are simply not in the training set rather than
            given special treatment.
          </p>
        </Section>

        <Section title="3. How a forecast is produced">
          <p>
            Every week, the model is given each country&apos;s historical series up to the most
            recent complete week and produces a probabilistic forecast for the next four weeks.
            The dashboard displays the central estimate on the map and 50%, 90% and 95%
            prediction intervals on the country pages. The pipeline updates automatically when
            the WHO publishes new weekly data.
          </p>
        </Section>

        <Section title="4. Evaluation">
          <p>
            Forecasts are scored with the <strong>weighted interval score (WIS)</strong>, a
            proper scoring rule for interval forecasts that rewards accurate central estimates
            and appropriately wide intervals. Because WIS is hard to interpret in absolute
            terms, we report it <strong>relative to a naive baseline</strong> (rWIS): the ratio
            of our WIS to the WIS of a simple seasonal-naive baseline computed on the same
            weeks. <em>rWIS &lt; 1 means the model beats the baseline; 1.0 means it matches
            it.</em>
          </p>
          <p>
            The headline validation is a <strong>head-to-head benchmark against Influcast</strong>
            , Italy&apos;s collaborative forecasting hub: for the 2025/26 season we compared our
            forecasts against the hub&apos;s nine participating models, its ensemble, and its
            naive quantile baseline across 16 weekly forecasting rounds. The season lies outside
            the fine-tuning data and post-dates the release of Chronos-2, making this a true
            out-of-sample test.
          </p>
          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm border border-gray-200 rounded-lg overflow-hidden bg-white">
              <thead>
                <tr className="bg-gray-100 text-left text-gray-600">
                  <th className="px-3 py-2 font-medium">Rank</th>
                  <th className="px-3 py-2 font-medium">Model</th>
                  <th className="px-3 py-2 font-medium text-right">Pairwise rWIS</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr
                    key={m.name}
                    className={`border-t border-gray-100 ${m.note === "us" ? "bg-blue-50" : ""}`}
                  >
                    <td className="px-3 py-1.5 text-gray-500">{m.rank}</td>
                    <td className="px-3 py-1.5">
                      {m.note === "us" ? (
                        <span className="font-semibold text-gray-900">{m.name}</span>
                      ) : (
                        <span className="text-gray-700">{m.name}</span>
                      )}
                      {m.note === "ensemble" && (
                        <span className="ml-2 text-xs text-gray-400">(9 models)</span>
                      )}
                      {m.note === "baseline" && (
                        <span className="ml-2 text-xs text-gray-400">(naive)</span>
                      )}
                    </td>
                    <td
                      className={`px-3 py-1.5 text-right tabular-nums ${
                        m.rwis < 1 ? "text-green-700 font-medium" : "text-gray-500"
                      }`}
                    >
                      {m.rwis.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-gray-400 mt-2">
              Italy, 2025/26 season, 16 weekly rounds, 4-week horizon. Lower is better; green =
              beats the naive baseline. The fine-tuned model finished second overall, ahead of
              the Influcast ensemble; zero-shot (unfine-tuned) Chronos-2 scored 1.01, i.e. no
              better than the baseline. Fine-tuning reduced rWIS by ~30% and MAPE from 23% to
              17%.
            </p>
          </div>
          <p>
            Per-country performance varies with data quality: well-monitored countries with long
            stable series are predicted well, while countries with sparse or irregular
            surveillance are much harder — and in a few cases the model adds no value over a
            naive baseline. We show per-country forecasts regardless, because even rough
            activity levels are better than nothing for countries that have no other
            operational forecast at all.
          </p>
        </Section>

        <Section title="5. Limitations">
          <ul className="list-disc pl-6 space-y-2">
            <li>
              <strong>Over-confident intervals.</strong> In the Italy benchmark, our 50%
              prediction intervals covered only ~24% of observed values. The central estimates
              are well calibrated relative to competing models, but the intervals are too
              narrow; treat them as indicative, not literal probabilities.
            </li>
            <li>
              <strong>The target is not influenza-specific.</strong> ILI and ARI counts include
              all acute respiratory illness — SARS-CoV-2, RSV, rhinovirus and others — not just
              influenza virus. Co-circulating viruses can therefore distort both the target and
              the forecasts.
            </li>
            <li>
              <strong>Surveillance artefacts.</strong> Reporting lags, mid-series indicator
              changes, and missing weeks are baked into the training data; the model learns
              them rather than correcting them.
            </li>
            <li>
              <strong>Single model, no ensemble.</strong> Purpose-built national systems
              (including Influcast&apos;s) benefit from model diversity; we do not yet combine
              multiple models or apply post-hoc calibration.
            </li>
            <li>
              <strong>Generalisation.</strong> The rigorous head-to-head validation is against
              one hub, in one country. Whether the same pattern holds against forecasting
              systems in other countries is an open question we intend to test.
            </li>
          </ul>
          <p className="text-sm text-gray-500 border-t border-gray-200 pt-4">
            This is a research tool. Forecasts are provided for comparison and exploration, not
            for operational decision-making.
          </p>
        </Section>
      </div>
    </main>
  );
}
