"use client";

import { useSearch } from "@/features/search/SearchContext";

type SearchBarProps = {
  className?: string;
};

export function SearchBar({ className = "" }: SearchBarProps) {
  const { query, setQuery, runSearch } = useSearch();

  return (
    <form
      className={["flex w-full gap-2", className].filter(Boolean).join(" ")}
      onSubmit={(e) => {
        e.preventDefault();
        runSearch();
      }}
    >
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Nhập text..."
        className="min-h-[52px] flex-1 rounded-2xl border border-surface-muted bg-white px-5 text-base text-ink shadow-card outline-none ring-accent/30 transition-shadow placeholder:text-ink-subtle focus:border-accent focus:ring-2"
        aria-label="Ô tìm kiếm"
      />
      <button
        type="submit"
        className="shrink-0 rounded-2xl bg-accent px-6 text-sm font-semibold text-white shadow-card transition hover:bg-accent-hover"
      >
        Gửi
      </button>
    </form>
  );
}
