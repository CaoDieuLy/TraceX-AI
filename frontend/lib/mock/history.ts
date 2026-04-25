export type SearchHistoryEntry = {
  id: string;
  query: string;
  at: string;
};

export const MOCK_SEARCH_HISTORY: SearchHistoryEntry[] = [
  { id: "h1", query: "người mặc áo trắng gần quầy", at: "2026-04-20 14:22" },
  { id: "h2", query: "xe đẩy y tế hành lang B", at: "2026-04-19 09:05" },
  { id: "h3", query: "nhóm 3 người lên thang máy", at: "2026-04-18 21:41" },
];
