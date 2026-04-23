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

import { GRID_BATCH_SIZE } from "@/lib/constants";
import type { VideoItem } from "@/lib/types/video";
import { searchVideos } from "@/lib/api";

type SearchContextValue = {
  query: string;
  setQuery: (value: string) => void;
  topK: number;
  setTopK: (value: number) => void;
  hasSearched: boolean;
  isLoading: boolean;
  error: string | null;
  hasMore: boolean;
  results: VideoItem[];
  gridPage: number;
  setGridPage: (page: number) => void;
  runSearch: () => Promise<void>;
  loadMore: () => Promise<boolean>;
  resetSearch: () => void;
};

const SearchContext = createContext<SearchContextValue | null>(null);

export function SearchProvider({ children }: { children: ReactNode }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(50);
  const [hasSearched, setHasSearched] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [results, setResults] = useState<VideoItem[]>([]);
  const [gridPage, setGridPage] = useState(0);

  const runSearch = useCallback(async () => {
    const trimmed = query.trim();
    if (!trimmed) {
      setError("Vui lòng nhập nội dung tìm kiếm.");
      setResults([]);
      setHasSearched(false);
      return;
    }

    setError(null);
    setHasSearched(true);
    setIsLoading(true);
    try {
      const requestLimit = Math.min(GRID_BATCH_SIZE, topK);
      const next = await searchVideos(trimmed, requestLimit, 0);
      setResults(next);
      setGridPage(0);
      setHasMore(next.length >= requestLimit && next.length < topK);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Không thể tải kết quả tìm kiếm.";
      setError(message);
      setResults([]);
      setHasMore(false);
    } finally {
      setIsLoading(false);
    }
  }, [query, topK]);

  const loadMore = useCallback(async (): Promise<boolean> => {
    const trimmed = query.trim();
    if (!trimmed || isLoading || !hasSearched || !hasMore) {
      return false;
    }

    const remaining = topK - results.length;
    if (remaining <= 0) {
      setHasMore(false);
      return false;
    }

    setIsLoading(true);
    setError(null);
    try {
      const requestLimit = Math.min(GRID_BATCH_SIZE, remaining);
      const next = await searchVideos(trimmed, requestLimit, results.length);
      if (!next.length) {
        setHasMore(false);
        return false;
      }
      setResults((prev) => [...prev, ...next]);
      setHasMore(next.length >= requestLimit && results.length + next.length < topK);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Không thể tải thêm kết quả.";
      setError(message);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [query, isLoading, hasSearched, hasMore, topK, results.length]);

  const resetSearch = useCallback(() => {
    setResults([]);
    setHasSearched(false);
    setError(null);
    setIsLoading(false);
    setHasMore(false);
    setGridPage(0);
  }, []);

  useEffect(() => {
    if (!hasSearched) {
      return;
    }
    const trimmed = query.trim();
    if (!trimmed) {
      return;
    }
    setIsLoading(true);
    setError(null);
    void searchVideos(trimmed, Math.min(GRID_BATCH_SIZE, topK), 0)
      .then((next) => {
        setResults(next);
        setGridPage(0);
        setHasMore(next.length >= Math.min(GRID_BATCH_SIZE, topK) && next.length < topK);
      })
      .catch((err) => {
        const message = err instanceof Error ? err.message : "Không thể tải kết quả tìm kiếm.";
        setError(message);
        setResults([]);
        setHasMore(false);
      })
      .finally(() => {
        setIsLoading(false);
      });
    // Only refetch for Top n change after user already searched.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topK, hasSearched]);

  const value = useMemo(
    () => ({
      query,
      setQuery,
      topK,
      setTopK,
      hasSearched,
      isLoading,
      error,
      hasMore,
      results,
      gridPage,
      setGridPage,
      runSearch,
      loadMore,
      resetSearch,
    }),
    [query, topK, hasSearched, isLoading, error, hasMore, results, gridPage, runSearch, loadMore, resetSearch],
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
