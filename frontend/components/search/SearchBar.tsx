"use client";

import { useSearch } from "@/features/search/SearchContext";

type SearchBarProps = {
  className?: string;
};

export function SearchBar({ className = "" }: SearchBarProps) {
  const { query, setQuery, runSearch, isLoading } = useSearch();

  return (
    <form
      className={["flex w-full gap-2", className].filter(Boolean).join(" ")}
      onSubmit={(e) => {
        e.preventDefault();
        void runSearch();
      }}
    >
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Nhập text..."
        className="min-h-[52px] flex-1 rounded-2xl border border-slate-200/90 bg-gradient-to-br from-white/95 to-blue-50/70 px-5 text-base font-medium text-slate-900 shadow-[0_12px_30px_rgba(15,23,42,0.08)] outline-none ring-sky-300/35 transition-[border-color,box-shadow] duration-200 placeholder:text-slate-400 focus:border-sky-400 focus:ring-2"
        aria-label="Ô tìm kiếm"
      />
      <button
        type="submit"
        disabled={isLoading}
        className="shrink-0 cursor-pointer rounded-2xl bg-[#0F172A] px-6 text-sm font-semibold text-white shadow-[0_14px_28px_rgba(15,23,42,0.24)] transition duration-200 hover:bg-[#1E293B] disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isLoading ? "Đang tìm..." : "Gửi"}
      </button>
    </form>
  );
}
