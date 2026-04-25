"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

const TOKEN_STORAGE_KEY = "mcpt_access_token";

function getApiBaseUrl(): string {
  const envBase = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").trim();
  if (envBase.startsWith("/")) {
    return envBase.replace(/\/$/, "");
  }
  return "";
}

type AuthResponse = {
  access_token: string;
  token_type: string;
  user: {
    id: number;
    email: string;
    full_name: string;
  };
};

async function readError(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        return payload.detail;
      }
      if (payload.detail !== undefined) {
        return String(payload.detail);
      }
    } catch {
      return `Request failed (${response.status})`;
    }
  }
  const text = (await response.text()).trim();
  return text || `Request failed (${response.status})`;
}

function normalizeEmail(username: string): string {
  const trimmed = username.trim();
  if (trimmed.includes("@")) {
    return trimmed;
  }
  return `${trimmed}@mcpt-app.com`;
}

export function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage(null);
    setError(null);

    const identifier = username.trim();
    if (!identifier || !password.trim()) {
      setError("Vui lòng nhập username và password.");
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identifier, password }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = (await response.json()) as AuthResponse;
      localStorage.setItem(TOKEN_STORAGE_KEY, payload.access_token);
      setMessage(`Đăng nhập thành công. Xin chào ${payload.user.full_name}!`);
      router.push("/home");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Đăng nhập thất bại.");
    } finally {
      setIsLoading(false);
    }
  }

  async function handleRegister() {
    setMessage(null);
    setError(null);

    const rawUsername = username.trim();
    if (!rawUsername || !password.trim()) {
      setError("Nhập username và password trước khi đăng ký.");
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${getApiBaseUrl()}/api/v1/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: normalizeEmail(rawUsername),
          full_name: rawUsername,
          password,
        }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = (await response.json()) as AuthResponse;
      localStorage.setItem(TOKEN_STORAGE_KEY, payload.access_token);
      setMessage(`Tạo tài khoản thành công: ${payload.user.email}`);
      router.push("/home");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Đăng ký thất bại.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface px-4 py-16">
      <div className="w-full max-w-md rounded-3xl border border-surface-muted bg-white p-8 shadow-elevated">
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-semibold text-ink">Đăng nhập</h1>
          <p className="mt-2 text-sm text-ink-secondary">Video search · mock auth</p>
        </div>

        <form className="flex flex-col gap-4" onSubmit={handleLogin}>
          <label className="flex flex-col gap-1.5 text-sm font-medium text-ink">
            Username
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              className="rounded-2xl border border-surface-muted px-4 py-3 text-base text-ink shadow-card outline-none ring-accent/25 focus:border-accent focus:ring-2"
              placeholder="username"
            />
          </label>
          <label className="flex flex-col gap-1.5 text-sm font-medium text-ink">
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              className="rounded-2xl border border-surface-muted px-4 py-3 text-base text-ink shadow-card outline-none ring-accent/25 focus:border-accent focus:ring-2"
              placeholder="••••••••"
            />
          </label>

          {message ? <p className="text-center text-sm text-ink-secondary">{message}</p> : null}
          {error ? <p className="text-center text-sm text-red-600">{error}</p> : null}

          <button
            type="submit"
            disabled={isLoading}
            className="mt-2 rounded-2xl bg-accent py-3 text-sm font-semibold text-white shadow-card transition hover:bg-accent-hover"
          >
            {isLoading ? "Đang xử lý..." : "Login"}
          </button>
          <button
            type="button"
            onClick={handleRegister}
            disabled={isLoading}
            className="rounded-2xl border border-surface-muted bg-white py-3 text-sm font-semibold text-ink shadow-card transition hover:bg-surface"
          >
            Đăng ký
          </button>
        </form>
      </div>
    </div>
  );
}
