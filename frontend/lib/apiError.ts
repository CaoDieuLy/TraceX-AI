/**
 * Parse JSON từ response thanh cong. Neu body la HTML (trang loi Next/proxy) thi bao loi ro rang.
 * Dung response.text() mot lan de tranh loi "Unexpected token '<'".
 */
export async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  const text = await response.text();
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("Phan hoi trong tu may chu.");
  }
  if (trimmed.startsWith("<")) {
    throw new Error(
      "May chu tra ve HTML thay vi JSON (thuong do sai URL, rewrite Next, hoac backend tra trang loi). " +
        "Hay chay lai: docker compose ... up -d --build frontend va kiem tra backend / rewrite.",
    );
  }
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    throw new Error("Phan hoi khong phai JSON hop le.");
  }
}

export function mapBackendErrorMessage(message: string): string {
  const normalized = String(message || "").toLowerCase();
  if (!normalized) return "Da xay ra loi. Vui long thu lai.";
  if (normalized.includes("authentication required") || normalized.includes("chua dang nhap")) {
    return "Vui long dang nhap de tiep tuc.";
  }
  if (normalized.includes("invalid") && normalized.includes("password")) {
    return "Thong tin dang nhap khong dung.";
  }
  if (normalized.includes("admin access required") || normalized.includes("khong co quyen")) {
    return "Ban khong co quyen thuc hien thao tac nay.";
  }
  if (normalized.includes("email already")) {
    return "Email da ton tai. Vui long dung email khac.";
  }
  if (normalized.includes("not found")) {
    return "Khong tim thay du lieu yeu cau.";
  }
  if (normalized.includes("downstream service error")) {
    return "Dich vu phu tro tam thoi loi. Vui long thu lai sau.";
  }
  if (normalized.includes("html")) {
    return "Loi ket noi API. Vui long thu lai sau.";
  }
  return message;
}

/**
 * Doc loi tu response (uu tien JSON FastAPI `detail`), khong goi .json() truc tiep tren HTML.
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
          typeof item === "object" && item && "msg" in item ? String((item as { msg: unknown }).msg) : String(item),
        );
        if (parts.length) return parts.join("; ");
      }
    } catch {
      // fall through
    }
  }

  if (text.startsWith("<")) {
    return "May chu tra ve trang HTML (404/405/loi Next hoac proxy). Hay rebuild frontend va kiem tra duong dan API.";
  }
  if (text.length > 0 && text.length < 400) {
    return text;
  }

  if (status === 401) return "Phien het han hoac chua dang nhap.";
  if (status === 403) return "Ban khong co quyen thuc hien thao tac nay (can ADMIN).";
  if (status === 404) return "Khong tim thay tai nguyen.";
  if (status === 405) return "Endpoint khong chap nhan phuong thuc nay (405).";
  if (status >= 500) return "Loi may chu. Vui long thu lai sau.";

  return mapBackendErrorMessage(`Yeu cau that bai (${status}).`);
}
