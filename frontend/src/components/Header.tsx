"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Header() {
  const pathname = usePathname();
  const isHome = pathname === "/";

  return (
    <header className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col md:flex-row md:items-center md:justify-between gap-1.5 md:gap-0 py-3 md:py-0 md:h-14">
        <Link href="/" className="text-base sm:text-lg font-semibold text-gray-900 hover:text-gray-700 transition-colors">
          Global Influenza Forecast
        </Link>
        <nav className="flex items-center justify-center md:justify-end gap-4 md:gap-6 text-sm">
          {[
            ["/", "Dashboard", false],
            ["/methodology", "Methodology", false],
            ["/about", "About", false],
            [
              "https://www.who.int/teams/global-influenza-programme/surveillance-and-monitoring/influenza-surveillance-outputs",
              "Data Source",
              true,
            ],
          ].map(([href, label, external]) => (
            <Link
              key={label as string}
              href={href as string}
              {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
              className={`transition-colors ${
                !external && pathname === href
                  ? "text-blue-600 font-medium"
                  : "text-gray-500 hover:text-gray-900"
              }`}
            >
              {label as string}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
