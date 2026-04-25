"use client";

import { Suspense, type ReactNode } from "react";

import { SearchBar } from "@/components/search/SearchBar";
import { TopKSelect } from "@/components/search/TopKSelect";
import { useSearch } from "@/features/search/SearchContext";

import { Sidebar } from "./Sidebar";

function SidebarFallback() {
  return (
    <aside className="flex h-screen w-[240px] shrink-0 animate-pulse border-r border-surface-muted bg-surface-card" />
  );
}

export function MainShell({ children }: { children: ReactNode }) {
  const { hasSearched } = useSearch();

  return (
    <div className="flex min-h-screen bg-surface">
      <Suspense fallback={<SidebarFallback />}>
        <Sidebar />
      </Suspense>
      <div className="flex min-w-0 flex-1 flex-col">
        <header
          className={
            hasSearched
              ? "sticky top-0 z-30 border-b border-surface-muted bg-surface/90 px-6 pb-4 pt-6 backdrop-blur-md"
              : "px-6 pb-2 pt-6"
          }
        >
          <div className="mx-auto flex max-w-6xl flex-col gap-4">
            <div className="flex flex-wrap items-start gap-4">
              <SearchBar className="min-w-[min(100%,280px)] flex-1" />
              {hasSearched ? (
                <div className="ml-auto shrink-0 pt-1">
                  <TopKSelect />
                </div>
              ) : null}
            </div>
          </div>
        </header>
        <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-6 py-8">{children}</main>
      </div>
    </div>
  );
}
