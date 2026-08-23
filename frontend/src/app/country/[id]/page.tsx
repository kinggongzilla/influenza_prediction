import { readdirSync } from "fs";
import path from "path";
import CountryDetails from "./CountryDetails";

// Static export (Cloudflare Pages): pre-render one page per country.
// Country codes are the ISO2 names of the JSON files in public/data/details/.
export function generateStaticParams() {
  const dir = path.join(process.cwd(), "public/data/details");
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => ({ id: f.replace(/\.json$/, "") }));
}

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <CountryDetails id={id} />;
}
