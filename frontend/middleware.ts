import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

/**
 * Tránh xung đột route: trang UI đặt tại /admin/users, còn POST/GET/PATCH /users proxy sang API gateway.
 * GET /users (bookmark cũ) chuyển hướng sang trang quản trị.
 * Lưu ý: không gọi fetch("GET /users") — dùng /api/users (rewrite) để luôn nhận JSON.
 */
export function middleware(request: NextRequest) {
  if (request.method === "GET" && request.nextUrl.pathname === "/users") {
    return NextResponse.redirect(new URL("/admin/users", request.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/users"],
};
