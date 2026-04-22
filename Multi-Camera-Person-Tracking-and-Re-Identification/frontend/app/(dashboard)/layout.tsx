import type { ReactNode } from "react";

import { MainShell } from "@/components/layout/MainShell";
import { SearchProvider } from "@/features/search/SearchContext";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <SearchProvider>
      <MainShell>{children}</MainShell>
    </SearchProvider>
  );
}
