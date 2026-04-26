"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

import { saveSession } from "@/lib/auth";
import { getApiBaseUrl } from "@/lib/api";
import { mapBackendErrorMessage, parseJsonOrThrow, readApiErrorMessage } from "@/lib/apiError";
import { useToast } from "@/components/ui/ToastProvider";

type AuthResponse = {
  access_token: string;
  token_type: string;
  user: {
    id: number;
    email: string;
    full_name?: string;
    role: "SUPER_ADMIN" | "ADMIN" | "USER";
    is_active: boolean;
  };
};

function friendlyErrorByStatus(status: number): string {
  if (status === 400) return "Thong tin dang nhap khong hop le.";
  if (status === 401) return "Email hoac mat khau khong dung.";
  if (status === 403) return "Tai khoan khong co quyen truy cap.";
  if (status === 404) return "Khong tim thay dich vu dang nhap. Vui long thu lai sau.";
  if (status >= 500) return "He thong dang ban. Vui long thu lai sau it phut.";
  return "Dang nhap that bai. Vui long thu lai.";
}

async function readError(response: Response): Promise<string> {
  try {
    return await readApiErrorMessage(response);
  } catch {
    return friendlyErrorByStatus(response.status);
  }
}

async function loginWithFallback(identifier: string, password: string): Promise<Response> {
  const baseUrl = getApiBaseUrl();
  const endpoints = [`${baseUrl}/auth/login`, `${baseUrl}/api/v1/auth/login`];
  let lastResponse: Response | null = null;

  for (const endpoint of endpoints) {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: identifier, identifier, password }),
    });
    if (response.ok) {
      return response;
    }
    lastResponse = response;
    // Fallback only when route truly missing.
    if (response.status !== 404) {
      return response;
    }
  }
  return lastResponse as Response;
}

export function LoginPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string }>({});

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFieldErrors({});

    const identifier = email.trim();
    const nextFieldErrors: { email?: string; password?: string } = {};
    if (!identifier) {
      nextFieldErrors.email = "Email khong duoc de trong.";
    }
    if (!password.trim()) {
      nextFieldErrors.password = "Password khong duoc de trong.";
    }
    if (Object.keys(nextFieldErrors).length > 0) {
      setFieldErrors(nextFieldErrors);
      showToast("Vui long kiem tra lai thong tin dang nhap.", "error");
      return;
    }

    setIsLoading(true);
    try {
      const response = await loginWithFallback(identifier, password);
      if (!response.ok) {
        const errorMessage = await readError(response);
        setFieldErrors({ email: "Thong tin dang nhap khong hop le.", password: "Thong tin dang nhap khong hop le." });
        throw new Error(errorMessage);
      }
      const payload = await parseJsonOrThrow<AuthResponse>(response);
      saveSession(payload.access_token, payload.user);
      showToast(`Dang nhap thanh cong. Xin chao ${payload.user.email}!`, "success");
      router.push("/home");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Dang nhap that bai.";
      showToast(mapBackendErrorMessage(message), "error");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div
      className="relative min-h-screen overflow-hidden"
      style={{
        backgroundImage: "url('/images/auth/img_login.png')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
      }}
    >
      <div className="absolute inset-0 bg-slate-950/50" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_22%,rgba(59,130,246,0.28),transparent_42%),radial-gradient(circle_at_80%_18%,rgba(14,165,233,0.18),transparent_40%)]" />

      <div className="relative mx-auto grid min-h-screen w-full max-w-7xl items-center gap-8 px-4 py-10 md:grid-cols-[1.2fr_0.8fr] md:px-10">
        <section className="rounded-3xl border border-white/15 bg-white/5 p-7 text-white shadow-elevated backdrop-blur-[1px] md:p-10">
          <p className="inline-flex rounded-full border border-white/25 bg-white/10 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-100">
            MCPT Intelligence
          </p>
          <h1 className="mt-5 text-4xl font-bold leading-tight text-white md:text-6xl">
            Tim nhanh hon.
            <br />
            Nhin ro hon.
          </h1>
          <p className="mt-6 max-w-2xl text-base leading-relaxed text-slate-100/90 md:text-lg">
            Nen tang quan ly va truy van video thong minh, toi uu cho van hanh noi bo va phan quyen an toan theo vai tro.
            Tat ca du lieu duoc tap trung trong mot giao dien hien dai, de su dung.
          </p>

          <div className="mt-8 grid gap-3 text-sm text-slate-100/95 sm:grid-cols-2">
            <div className="rounded-2xl border border-white/15 bg-white/10 px-4 py-3">Tra cuu video theo text theo ngu canh.</div>
            <div className="rounded-2xl border border-white/15 bg-white/10 px-4 py-3">Quan ly user voi RBAC ro rang, an toan.</div>
            <div className="rounded-2xl border border-white/15 bg-white/10 px-4 py-3">Theo doi ket qua tap trung, thao tac nhanh.</div>
            <div className="rounded-2xl border border-white/15 bg-white/10 px-4 py-3">San sang mo rong cho he thong AI service.</div>
          </div>
        </section>

        <section className="flex items-center justify-center md:justify-end">
          <div className="w-full max-w-md rounded-3xl border border-white/40 bg-white/92 p-8 shadow-elevated backdrop-blur-md md:p-9">
            <div className="mb-8 text-center">
              <h2 className="text-2xl font-semibold text-ink">Log in</h2>
              <p className="mt-2 text-sm text-ink-secondary">Su dung email va password de tiep tuc</p>
            </div>

            <form className="flex flex-col gap-4" onSubmit={handleLogin}>
              <label className="flex flex-col gap-1.5 text-sm font-medium text-ink">
                Email
                <input
                  type="email"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                    if (fieldErrors.email) {
                      setFieldErrors((prev) => ({ ...prev, email: undefined }));
                    }
                  }}
                  autoComplete="username"
                  className={[
                    "rounded-2xl border px-4 py-3 text-base text-ink shadow-card outline-none ring-accent/25 focus:ring-2",
                    fieldErrors.email
                      ? "border-red-300 bg-red-50 focus:border-red-400"
                      : "border-surface-muted focus:border-accent",
                  ].join(" ")}
                  placeholder="name@company.com"
                  aria-invalid={Boolean(fieldErrors.email)}
                />
                {fieldErrors.email ? <span className="text-xs text-red-600">{fieldErrors.email}</span> : null}
              </label>
              <label className="flex flex-col gap-1.5 text-sm font-medium text-ink">
                Password
                <input
                  type="password"
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    if (fieldErrors.password) {
                      setFieldErrors((prev) => ({ ...prev, password: undefined }));
                    }
                  }}
                  autoComplete="current-password"
                  className={[
                    "rounded-2xl border px-4 py-3 text-base text-ink shadow-card outline-none ring-accent/25 focus:ring-2",
                    fieldErrors.password
                      ? "border-red-300 bg-red-50 focus:border-red-400"
                      : "border-surface-muted focus:border-accent",
                  ].join(" ")}
                  placeholder="••••••••"
                  aria-invalid={Boolean(fieldErrors.password)}
                />
                {fieldErrors.password ? <span className="text-xs text-red-600">{fieldErrors.password}</span> : null}
              </label>

              <button
                type="submit"
                disabled={isLoading}
                className="mt-2 rounded-2xl bg-accent py-3 text-sm font-semibold text-white shadow-card transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isLoading ? "Dang xu ly..." : "Log in"}
              </button>

              <button
                type="button"
                onClick={() => showToast("Tinh nang forgot password se duoc bo sung sau.", "info")}
                className="rounded-2xl border border-surface-muted bg-white py-3 text-sm font-semibold text-ink-secondary transition hover:bg-surface hover:text-ink"
              >
                Forgot password
              </button>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
}
