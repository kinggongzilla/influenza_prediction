"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Header() {
  const pathname = usePathname();
  const isHome = pathname === "/";

  return (
    <header className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link href="/" className="text-lg font-semibold text-gray-900 hover:text-gray-700 transition-colors">
          Global Influenza Forecast
        </Link>
        <nav className="flex items-center gap-6 text-sm">
          <Link
            href="/"
            className={`transition-colors ${isHome ? "text-blue-600 font-medium" : "text-gray-500 hover:text-gray-900"}`}
          >
            Dashboard
          </Link>
          <a
            href="https://github.com/WHO-Collaboratory"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-500 hover:text-gray-900 transition-colors"
          >
            Data Source
          </a>
        </nav>
      </div>
    </header>
  );
}
