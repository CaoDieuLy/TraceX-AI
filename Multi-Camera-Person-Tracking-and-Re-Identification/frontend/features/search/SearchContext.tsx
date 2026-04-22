"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import type { VideoItem } from "@/lib/types/video";
import { mockSearchResults } from "@/lib/mock/videos";

type SearchContextValue = {
  query: string;
  setQuery: (value: string) => void;
  topK: number;
  setTopK: (value: number) => void;
  hasSearched: boolean;
  results: VideoItem[];
  gridPage: number;
  setGridPage: (page: number) => void;
  runSearch: () => void;
  resetSearch: () => void;
};

const SearchContext = createContext<SearchContextValue | null>(null);

export function SearchProvider({ children }: { children: ReactNode }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(50);
  const [hasSearched, setHasSearched] = useState(false);
  const [results, setResults] = useState<VideoItem[]>([]);
  const [gridPage, setGridPage] = useState(0);

  const runSearch = useCallback(() => {
    const next = mockSearchResults(query, topK);
    setResults(next);
    setHasSearched(true);
    setGridPage(0);
  }, [query, topK]);

  const resetSearch = useCallback(() => {
    setResults([]);
    setHasSearched(false);
    setGridPage(0);
  }, []);

  useEffect(() => {
    if (!hasSearched) {
      return;
    }
    setResults(mockSearchResults(query, topK));
    setGridPage(0);
    // Intentionally omit `query`: only refetch when Top K changes or search mode toggles (not on every keystroke).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topK, hasSearched]);

  const value = useMemo(
    () => ({
      query,
      setQuery,
      topK,
      setTopK,
      hasSearched,
      results,
      gridPage,
      setGridPage,
      runSearch,
      resetSearch,
    }),
    [query, topK, hasSearched, results, gridPage, runSearch, resetSearch],
  );

  return <SearchContext.Provider value={value}>{children}</SearchContext.Provider>;
}

export function useSearch() {
  const ctx = useContext(SearchContext);
  if (!ctx) {
    throw new Error("useSearch must be used within SearchProvider");
  }
  return ctx;
}
