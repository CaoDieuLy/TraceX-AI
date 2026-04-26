"use client";

import { TOP_K_OPTIONS } from "@/lib/constants";
import { useSearch } from "@/features/search/SearchContext";

export function TopKSelect() {
  const { topK, setTopK } = useSearch();

  return (
    <label className="flex shrink-0 items-center gap-2 text-sm text-ink-secondary">
      <span className="whitespace-nowrap font-medium">Top n</span>
      <select
        value={topK}
        onChange={(e) => setTopK(Number(e.target.value))}
        className="rounded-xl border border-surface-muted bg-white px-3 py-2 text-sm font-medium text-ink shadow-card outline-none ring-accent/20 focus:ring-2"
      >
        {TOP_K_OPTIONS.map((k) => (
          <option key={k} value={k}>
            Top {k}
          </option>
        ))}
      </select>
    </label>
  );
}
