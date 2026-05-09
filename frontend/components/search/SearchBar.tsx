"use client";

import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";

import { useSearch } from "@/features/search/SearchContext";
import { LOCATION_OPTIONS, summarizeSelectedLocations } from "@/lib/config";

type SearchBarProps = {
  className?: string;
};

export function SearchBar({ className = "" }: SearchBarProps) {
  const {
    query,
    setQuery,
    image,
    setImage,
    runSearch,
    isLoading,
    filters,
    setLocationIds,
    setTimeFrom,
    setTimeTo,
    clearFilters,
    filterError,
  } = useSearch();
  const [locationOpen, setLocationOpen] = useState(false);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [locationKeyword, setLocationKeyword] = useState("");
  const [imageError, setImageError] = useState<string | null>(null);
  const locationRef = useRef<HTMLDivElement | null>(null);
  const imageInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    function onClickOutside(event: MouseEvent) {
      if (!locationRef.current) return;
      if (!locationRef.current.contains(event.target as Node)) {
        setLocationOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const filteredLocations = useMemo(() => {
    const keyword = locationKeyword.trim().toLowerCase();
    if (!keyword) return LOCATION_OPTIONS;
    return LOCATION_OPTIONS.filter((option) => {
      const labels = [option.label, ...option.keywords].join(" ").toLowerCase();
      return labels.includes(keyword);
    });
  }, [locationKeyword]);

  const selectedSummary = summarizeSelectedLocations(filters.locationIds);
  const selectedCount = filters.locationIds.length;
  const hasActiveFilters = selectedCount > 0 || Boolean(filters.timeFrom || filters.timeTo);

  function toggleLocation(locationId: string) {
    const selected = new Set(filters.locationIds);
    if (selected.has(locationId)) {
      selected.delete(locationId);
    } else {
      selected.add(locationId);
    }
    setLocationIds(Array.from(selected));
  }

  function clearImage() {
    setImage(null);
    setImageError(null);
    if (imageInputRef.current) imageInputRef.current.value = "";
  }

  function handleImageChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setImageError("Chỉ hỗ trợ tệp ảnh.");
      event.target.value = "";
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setImageError("Ảnh tối đa 5MB.");
      event.target.value = "";
      return;
    }
    setImage({ name: file.name, size: file.size, type: file.type || "image/jpeg", previewUrl: URL.createObjectURL(file) });
    setImageError(null);
  }

  return (
    <form
      className={["flex w-full flex-col gap-2 dark:[color-scheme:dark]", className].filter(Boolean).join(" ")}
      onSubmit={(e) => { e.preventDefault(); void runSearch(); }}
    >
      {/* ── Main search bar ── */}
      <div className="flex min-h-[58px] w-full items-center overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_8px_24px_rgba(15,23,42,0.08)] transition-colors dark:border-slate-800 dark:bg-slate-950 dark:shadow-[0_10px_30px_rgba(0,0,0,0.35)]">
        {/* Search icon */}
        <div className="flex h-14 w-14 shrink-0 items-center justify-center border-r border-slate-100 text-slate-500 dark:border-slate-800 dark:text-slate-400">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="6" />
            <path d="m16 16 4 4" />
          </svg>
        </div>

        {/* Text input */}
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Nhập mô tả người hoặc hành vi cần tìm..."
          className="min-h-[56px] min-w-0 flex-1 border-0 bg-white px-4 text-sm font-medium text-slate-900 outline-none placeholder:text-slate-400 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-500"
          aria-label="Ô tìm kiếm"
        />

        {/* Hidden file input */}
        <input ref={imageInputRef} type="file" accept="image/*" className="hidden" onChange={handleImageChange} />

        {/* Image button (desktop) */}
        <button
          type="button"
          onClick={() => imageInputRef.current?.click()}
          className={[
            "hidden min-h-[44px] shrink-0 items-center justify-center border-l border-slate-100 px-3 text-sm font-semibold transition dark:border-slate-800 sm:flex",
            image ? "text-blue-700 dark:text-blue-300" : "text-slate-600 hover:text-slate-950 dark:text-slate-300 dark:hover:text-white",
          ].join(" ")}
          aria-label="Tải ảnh tìm kiếm"
          title="Tải ảnh tìm kiếm"
        >
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="3" y="5" width="18" height="14" rx="2" />
            <circle cx="8.5" cy="10" r="1.5" />
            <path d="m21 15-4.5-4.5L9 18" />
          </svg>
        </button>

        {/* Filter toggle button (desktop) */}
        <button
          type="button"
          onClick={() => setFiltersOpen((prev) => !prev)}
          className={[
            "hidden min-h-[44px] shrink-0 items-center gap-2 border-l border-slate-100 px-4 text-sm font-semibold transition dark:border-slate-800 sm:flex",
            filtersOpen || hasActiveFilters ? "text-blue-700 dark:text-blue-300" : "text-slate-600 hover:text-slate-950 dark:text-slate-300 dark:hover:text-white",
          ].join(" ")}
          aria-expanded={filtersOpen}
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M4 7h10" /><path d="M18 7h2" />
            <path d="M4 17h2" /><path d="M10 17h10" />
            <circle cx="16" cy="7" r="2" /><circle cx="8" cy="17" r="2" />
          </svg>
          Bộ lọc
        </button>

        {/* Submit button */}
        <button
          type="submit"
          disabled={isLoading}
          className="mr-2 flex min-h-[44px] shrink-0 cursor-pointer items-center gap-2 rounded-2xl bg-blue-600 px-4 text-sm font-bold text-white shadow-[0_12px_22px_rgba(37,99,235,0.32)] transition duration-200 hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50 sm:px-5"
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="m22 2-7 20-4-9-9-4Z" />
            <path d="M22 2 11 13" />
          </svg>
          {isLoading ? "Đang tìm..." : "Gửi"}
        </button>
      </div>

      {/* ── Mobile action row ── */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Image button (mobile) */}
        <button
          type="button"
          onClick={() => imageInputRef.current?.click()}
          className="flex min-h-[42px] items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-blue-200 hover:text-blue-700 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-blue-700 dark:hover:text-blue-300 sm:hidden"
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="3" y="5" width="18" height="14" rx="2" />
            <circle cx="8.5" cy="10" r="1.5" />
            <path d="m21 15-4.5-4.5L9 18" />
          </svg>
          Ảnh
        </button>

        {/* Image preview chip */}
        {image ? (
          <div className="flex max-w-full items-center gap-3 rounded-xl border border-blue-200 bg-blue-50 px-3 py-2 text-sm font-semibold text-blue-900 dark:border-blue-500/40 dark:bg-blue-500/10 dark:text-blue-100">
            <span
              className="h-10 w-10 rounded-lg bg-cover bg-center"
              style={{ backgroundImage: `url(${image.previewUrl})` }}
              aria-hidden="true"
            />
            <span className="max-w-[220px] truncate">{image.name}</span>
            <button type="button" onClick={clearImage} className="text-xs font-black text-blue-700 hover:text-blue-900 dark:text-blue-200">
              Xóa
            </button>
          </div>
        ) : null}
      </div>

      {/* Filter toggle (mobile) */}
      <button
        type="button"
        onClick={() => setFiltersOpen((prev) => !prev)}
        className="flex min-h-[42px] items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-blue-200 hover:text-blue-700 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-blue-700 dark:hover:text-blue-300 sm:hidden"
        aria-expanded={filtersOpen}
      >
        Bộ lọc
      </button>

      {imageError ? <p className="text-xs font-medium text-red-600">{imageError}</p> : null}

      {/* ── Filter panel (collapsible) ── */}
      <div className={["flex flex-wrap items-center gap-2", filtersOpen ? "flex" : "hidden"].join(" ")}>
        {/* Location dropdown */}
        <div className="relative" ref={locationRef}>
          <button
            type="button"
            className="min-h-[42px] cursor-pointer rounded-xl border border-slate-200 bg-white px-4 text-sm font-semibold text-slate-700 shadow-sm transition duration-200 hover:border-blue-300 hover:text-slate-900 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-200 dark:hover:border-blue-700 dark:hover:text-white"
            onClick={() => setLocationOpen((prev) => !prev)}
          >
            {selectedCount ? `${selectedSummary} (${selectedCount})` : "Vị trí"}
          </button>
          {locationOpen ? (
            <div className="absolute left-0 top-[calc(100%+8px)] z-40 w-[320px] rounded-2xl border border-slate-200 bg-white p-3 shadow-[0_18px_36px_rgba(15,23,42,0.16)] dark:border-slate-800 dark:bg-slate-900 dark:shadow-[0_20px_42px_rgba(0,0,0,0.45)]">
              <input
                type="text"
                value={locationKeyword}
                onChange={(event) => setLocationKeyword(event.target.value)}
                placeholder="Tìm vị trí..."
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 outline-none ring-blue-300/35 focus:border-blue-400 focus:ring-2 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-500"
              />
              <div className="mt-2 max-h-56 space-y-1 overflow-y-auto pr-1">
                {filteredLocations.map((option) => {
                  const checked = filters.locationIds.includes(option.id);
                  return (
                    <label
                      key={option.id}
                      className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-2 text-sm text-slate-700 transition hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-800"
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleLocation(option.id)}
                        className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="font-medium">{option.label}</span>
                    </label>
                  );
                })}
              </div>
              <button
                type="button"
                onClick={() => setLocationIds([])}
                className="mt-2 text-xs font-semibold text-blue-700 hover:text-blue-800"
              >
                Bỏ chọn vị trí
              </button>
            </div>
          ) : null}
        </div>

        <input
          type="datetime-local"
          value={filters.timeFrom}
          onChange={(event) => setTimeFrom(event.target.value)}
          className="min-h-[42px] rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-700 shadow-sm outline-none ring-blue-300/35 focus:border-blue-400 focus:ring-2 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-200"
          aria-label="Thời gian bắt đầu"
        />
        <input
          type="datetime-local"
          value={filters.timeTo}
          onChange={(event) => setTimeTo(event.target.value)}
          className="min-h-[42px] rounded-xl border border-slate-200 bg-white px-3 text-sm text-slate-700 shadow-sm outline-none ring-blue-300/35 focus:border-blue-400 focus:ring-2 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-200"
          aria-label="Thời gian kết thúc"
        />
        <button
          type="button"
          onClick={clearFilters}
          className="min-h-[42px] cursor-pointer rounded-xl border border-slate-200 bg-white px-4 text-sm font-semibold text-slate-600 transition duration-200 hover:border-slate-300 hover:text-slate-800 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300 dark:hover:border-slate-700 dark:hover:text-white"
        >
          Xóa lọc
        </button>
      </div>

      {filterError ? <p className="text-xs font-medium text-red-600">{filterError}</p> : null}
    </form>
  );
}
