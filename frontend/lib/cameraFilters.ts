export type LocationOption = {
  id: string;
  label: string;
  cameraIds: string[];
  keywords: string[];
};

// Derived from backend/config/camera_topology.json (grouped by area_group/area_name for user-friendly labels).
export const LOCATION_OPTIONS: LocationOption[] = [
  { id: "entrance", label: "Khu cổng vào", cameraIds: ["cam01", "cam02"], keywords: ["cong", "entrance", "drop-off"] },
  { id: "parking", label: "Bãi xe", cameraIds: ["cam03", "cam04", "cam05", "cam06"], keywords: ["bai xe", "parking"] },
  { id: "lobby", label: "Sảnh chính", cameraIds: ["cam07", "cam08", "cam12"], keywords: ["sanh", "lobby"] },
  { id: "reception", label: "Lễ tân & đăng ký", cameraIds: ["cam09", "cam10"], keywords: ["le tan", "dang ky", "reception"] },
  { id: "lobby_waiting", label: "Khu chờ sảnh", cameraIds: ["cam11"], keywords: ["khu cho", "waiting"] },
  { id: "corridor_1f", label: "Hành lang tầng 1", cameraIds: ["cam13", "cam14", "cam15", "cam16", "cam17"], keywords: ["hanh lang", "tang 1", "corridor"] },
  { id: "clinic", label: "Khu khám bệnh", cameraIds: ["cam18", "cam19", "cam20", "cam21", "cam22"], keywords: ["kham benh", "clinic"] },
  { id: "lab", label: "Khu xét nghiệm", cameraIds: ["cam23", "cam24", "cam25"], keywords: ["xet nghiem", "lab"] },
  { id: "imaging", label: "Khu chẩn đoán hình ảnh", cameraIds: ["cam26", "cam27", "cam28"], keywords: ["chan doan", "imaging"] },
  { id: "pharmacy", label: "Nhà thuốc", cameraIds: ["cam29", "cam30"], keywords: ["nha thuoc", "pharmacy"] },
  { id: "vertical", label: "Thang máy & cầu thang", cameraIds: ["cam31", "cam32", "cam33", "cam34"], keywords: ["thang may", "cau thang", "vertical"] },
  { id: "corridor_2f", label: "Nút giao tầng 2", cameraIds: ["cam35"], keywords: ["nut giao", "tang 2"] },
  { id: "ward", label: "Khu nội trú", cameraIds: ["cam36", "cam37", "cam38", "cam39", "cam40", "cam41", "cam42"], keywords: ["noi tru", "ward"] },
  { id: "ward_side", label: "Hành lang phụ & kho tầng 2", cameraIds: ["cam43", "cam44", "cam45"], keywords: ["hanh lang phu", "kho", "exit"] },
  { id: "emergency", label: "Khu cấp cứu", cameraIds: ["cam46", "cam47", "cam48", "cam49", "cam50"], keywords: ["cap cuu", "emergency"] },
];

export function mapLocationIdsToCameraIds(locationIds: string[]): string[] {
  const selected = new Set(locationIds);
  const merged = LOCATION_OPTIONS.flatMap((option) => (selected.has(option.id) ? option.cameraIds : []));
  return Array.from(new Set(merged));
}

export function summarizeSelectedLocations(locationIds: string[]): string {
  if (!locationIds.length) return "Vị trí";
  const selectedOptions = LOCATION_OPTIONS.filter((option) => locationIds.includes(option.id));
  if (!selectedOptions.length) return "Vị trí";
  if (selectedOptions.length === 1) return selectedOptions[0].label;
  return `${selectedOptions[0].label} +${selectedOptions.length - 1}`;
}
