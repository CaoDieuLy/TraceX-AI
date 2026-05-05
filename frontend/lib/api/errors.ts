import { loadAccessToken } from "@/lib/auth";

/**
 * Parse JSON từ response thành công. Nếu body là HTML (trang lỗi Next/proxy) thì báo lỗi rõ ràng.
 * Dùng response.text() một lần để tránh lỗi "Unexpected token '<'".
 */
export async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const text = await response.text();
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("Phản hồi trống từ máy chủ.");
  }
  if (trimmed.startsWith("<")) {
    throw new Error(
      "Máy chủ trả về HTML thay vì JSON (thường do sai URL, rewrite Next, hoặc backend trả trang lỗi). " +
        "Hãy build lại: docker compose ... up -d --build frontend và kiểm tra backend / rewrite."
    );
  }
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    throw new Error("Phản hồi không phải JSON hợp lệ.");
  }
}

export function mapBackendErrorMessage(message: string): string {
  const normalized = String(message || "").toLowerCase();
  if (!normalized) return "Đã xảy ra lỗi. Vui lòng thử lại.";
  if (normalized.includes("authentication required") || normalized.includes("chưa đăng nhập")) {
    return "Vui lòng đăng nhập để tiếp tục.";
  }
  if (normalized.includes("invalid") && normalized.includes("password")) {
    return "Thông tin đăng nhập không đúng.";
  }
  if (normalized.includes("admin access required") || normalized.includes("không có quyền")) {
    return "Bạn không có quyền thực hiện thao tác này.";
  }
  if (normalized.includes("email already")) {
    return "Email đã tồn tại. Vui lòng dùng email khác.";
  }
  if (normalized.includes("not found")) {
    return "Không tìm thấy dữ liệu yêu cầu.";
  }
  if (normalized.includes("downstream service error")) {
    return "Dịch vụ phụ trợ tạm thời lỗi. Vui lòng thử lại sau.";
  }
  if (normalized.includes("html")) {
    return "Lỗi kết nối API. Vui lòng thử lại sau.";
  }
  return message;
}

/**
 * Đọc lỗi từ response (ưu tiên JSON FastAPI `detail`), không gọi .json() trực tiếp trên HTML.
 */
export async function readApiErrorMessage(response: Response): Promise<string> {
  const status = response.status;
  const text = (await response.text()).trim();

  if (text.startsWith("{") || text.startsWith("[")) {
    try {
      const body = JSON.parse(text) as { detail?: unknown };
      const detail = body.detail;
      if (typeof detail === "string" && detail.trim()) return detail;
      if (Array.isArray(detail)) {
        const parts = detail.map((item) =>
          typeof item === "object" && item && "msg" in item ? String((item as { msg: unknown }).msg) : String(item)
        );
        if (parts.length) return parts.join("; ");
      }
    } catch {
      // fall through
    }
  }

  if (text.startsWith("<")) {
    return "Máy chủ trả về trang HTML (404/405/lỗi Next hoặc proxy). Hãy rebuild frontend và kiểm tra đường dẫn API.";
  }
  if (text.length > 0 && text.length < 400) {
    return text;
  }

  if (status === 401) return "Phiên hết hạn hoặc chưa đăng nhập.";
  if (status === 403) return "Bạn không có quyền thực hiện thao tác này (cần ADMIN).";
  if (status === 404) return "Không tìm thấy tài nguyên.";
  if (status === 405) return "Endpoint không chấp nhận phương thức này (405).";
  if (status >= 500) return "Lỗi máy chủ. Vui lòng thử lại sau.";

  return mapBackendErrorMessage(`Yêu cầu thất bại (${status}).`);
}
