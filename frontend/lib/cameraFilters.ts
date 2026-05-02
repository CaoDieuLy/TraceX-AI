export type LocationOption = {
  id: string;
  label: string;
  cameraIds: string[];
  keywords: string[];
};

// Derived from backend/config/camera_topology.json (grouped by area_group/area_name for user-friendly labels).
export const LOCATION_OPTIONS: LocationOption[] = [
  { id: "entrance",    label: "Khu cổng vào",                  cameraIds: ["cam_01", "cam_02"],                                               keywords: ["cong", "entrance", "drop-off"] },
  { id: "parking",    label: "Bãi xe",                         cameraIds: ["cam_03", "cam_04", "cam_05", "cam_06"],                           keywords: ["bai xe", "parking"] },
  { id: "lobby",      label: "Sảnh chính",                     cameraIds: ["cam_07", "cam_08", "cam_12"],                                     keywords: ["sanh", "lobby"] },
  { id: "reception",  label: "Lễ tân & đăng ký",               cameraIds: ["cam_09", "cam_10"],                                               keywords: ["le tan", "dang ky", "reception"] },
  { id: "lobby_waiting", label: "Khu chờ sảnh",                cameraIds: ["cam_11"],                                                         keywords: ["khu cho", "waiting"] },
  { id: "corridor_1f", label: "Hành lang tầng 1",              cameraIds: ["cam_13", "cam_14", "cam_15", "cam_16", "cam_17"],                 keywords: ["hanh lang", "tang 1", "corridor"] },
  { id: "clinic",     label: "Khu khám bệnh",                  cameraIds: ["cam_18", "cam_19", "cam_20", "cam_21", "cam_22"],                 keywords: ["kham benh", "clinic"] },
  { id: "lab",        label: "Khu xét nghiệm",                 cameraIds: ["cam_23", "cam_24", "cam_25"],                                     keywords: ["xet nghiem", "lab"] },
  { id: "imaging",    label: "Khu chẩn đoán hình ảnh",         cameraIds: ["cam_26", "cam_27", "cam_28"],                                     keywords: ["chan doan", "imaging"] },
  { id: "pharmacy",   label: "Nhà thuốc",                      cameraIds: ["cam_29", "cam_30"],                                               keywords: ["nha thuoc", "pharmacy"] },
  { id: "vertical",   label: "Thang máy & cầu thang",          cameraIds: ["cam_31", "cam_32", "cam_33", "cam_34"],                           keywords: ["thang may", "cau thang", "vertical"] },
  { id: "corridor_2f", label: "Nút giao tầng 2",               cameraIds: ["cam_35"],                                                         keywords: ["nut giao", "tang 2"] },
  { id: "ward",       label: "Khu nội trú",                    cameraIds: ["cam_36", "cam_37", "cam_38", "cam_39", "cam_40", "cam_41", "cam_42"], keywords: ["noi tru", "ward"] },
  { id: "ward_side",  label: "Hành lang phụ & kho tầng 2",     cameraIds: ["cam_43", "cam_44", "cam_45"],                                     keywords: ["hanh lang phu", "kho", "exit"] },
  { id: "emergency",  label: "Khu cấp cứu",                    cameraIds: ["cam_46", "cam_47", "cam_48", "cam_49", "cam_50"],                 keywords: ["cap cuu", "emergency"] },
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
