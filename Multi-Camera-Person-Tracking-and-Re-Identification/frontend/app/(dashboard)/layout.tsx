import type { ReactNode } from "react";

import { MainShell } from "@/components/layout/MainShell";
import { AuthGate } from "@/features/auth/AuthGate";
import { SearchProvider } from "@/features/search/SearchContext";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <AuthGate>
      <SearchProvider>
        <MainShell>{children}</MainShell>
      </SearchProvider>
    </AuthGate>
  );
}
