export interface SearchHistoryEntry {
  id: string;
  timestamp: string;
  query: string;
  videoName: string;
  status: "completed" | "processing" | "failed";
}

export const MOCK_SEARCH_HISTORY: SearchHistoryEntry[] = [
  {
    id: "1",
    timestamp: "2024-01-15 10:30",
    query: "Người đi bộ qua đường",
    videoName: "intersection_cam1.mp4",
    status: "completed",
  },
  {
    id: "2",
    timestamp: "2024-01-15 09:15",
    query: "Xe máy chạy quá tốc độ",
    videoName: "highway_traffic.mp4",
    status: "completed",
  },
  {
    id: "3",
    timestamp: "2024-01-14 16:45",
    query: "Hành vi khả nghi",
    videoName: "parking_lot.mp4",
    status: "processing",
  },
  {
    id: "4",
    timestamp: "2024-01-14 14:20",
    query: "Đối tượng bỏ trốn",
    videoName: "warehouse_door.mp4",
    status: "failed",
  },
  {
    id: "5",
    timestamp: "2024-01-13 11:00",
    query: "Xe tải dừng đỗ sai quy định",
    videoName: "loading_zone.mp4",
    status: "completed",
  },
];
