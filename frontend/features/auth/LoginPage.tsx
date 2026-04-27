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
        backgroundImage: "url('/images/auth/img_backgroud5.png')",
        backgroundSize: "cover",
        backgroundPosition: "center top",
        backgroundRepeat: "no-repeat",
      }}
    >
      <div className="absolute inset-0 bg-slate-900/12" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_22%_30%,rgba(255,255,255,0.34),transparent_48%),radial-gradient(circle_at_82%_20%,rgba(174,213,255,0.26),transparent_45%)]" />


      <div className="relative mx-auto grid min-h-screen w-full max-w-7xl items-center gap-8 px-4 py-8 md:grid-cols-[1.22fr_0.78fr] md:px-10">
        <section className="rounded-3xl bg-slate-900/12 p-6 pt-20 text-white backdrop-blur-[2px] md:p-10 md:pt-36">
          <div className="flex items-center gap-3">
            <img src="/images/auth/img_icon_se.png" alt="logo" className="h-11 w-11 rounded-xl object-contain" />
            <div>
              <p className="text-3xl font-bold leading-tight text-[#F6FAFF] drop-shadow-[0_2px_6px_rgba(10,30,60,0.42)]">
                VLyMVision
              </p>
              <p className="text-sm text-[#E8F2FF]/90">AI-Powered CCTV Platform</p>
            </div>
          </div>

          <h1 className="mt-7 max-w-3xl text-4xl font-bold leading-tight md:text-6xl">
            <span className="text-[#F6FAFF] drop-shadow-[0_2px_6px_rgba(10,30,60,0.42)]">Giam sat thong minh,</span>
            <br />
            <span className="text-[#5DD4FF] drop-shadow-[0_2px_10px_rgba(34,200,255,0.45)]">an toàn vượt trội.</span>
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-[#E8F2FF] md:text-lg">
            Nen tang CCTV tich hop AI giup ban giam sat, phat hien va phan ung tuc thi moi su kien.
          </p>

          <div className="mt-8 max-w-xl space-y-3">
            <div className="flex items-start gap-3">
              <img src="/images/auth/img_icon_Ai.png" alt="AI" className="mt-0.5 h-12 w-12 rounded-lg object-contain" />
              <div>
                <p className="text-xl font-semibold text-[#F5FAFF]">AI Thong minh</p>
                <p className="text-sm text-[#DFECFF]">Phat hien khuon mat, hanh vi bat thuong va canh bao theo thoi gian thuc.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <img src="/images/auth/img_icon_cam.png" alt="Camera" className="mt-0.5 h-12 w-12 rounded-lg object-contain" />
              <div>
                <p className="text-xl font-semibold text-[#F5FAFF]">Giam sat toan dien</p>
                <p className="text-sm text-[#DFECFF]">Xem live, xem lai, quan ly nhieu camera moi luc moi noi.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <img src="/images/auth/img_icon_se2.png" alt="Search" className="mt-0.5 h-12 w-12 rounded-lg object-contain" />
              <div>
                <p className="text-xl font-semibold text-[#F5FAFF]">Tim nguoi theo mo ta</p>
                <p className="text-sm text-[#DFECFF]">Tim kiem nguoi qua dac diem hoac trang phuc.</p>
              </div>
            </div>
          </div>

          <div className="mt-7 flex flex-wrap items-center gap-6 text-sm text-[#EAF3FF]">
            <div className="flex items-center gap-2">
              <img src="/images/auth/img_icon_se2.png" alt="safe" className="h-5 w-5 rounded-md border border-white/70 bg-white/90 p-0.5 shadow-sm object-contain" />
              <span>Bao mat cao</span>
            </div>
            <div className="flex items-center gap-2">
              <img src="/images/auth/img_icon_cloud.png" alt="cloud" className="h-5 w-5 rounded-md border border-white/70 bg-white/90 p-0.5 shadow-sm object-contain" />
              <span>Luu tru linh hoat</span>
            </div>
            <div className="flex items-center gap-2">
              <img src="/images/auth/img_icon_thunder.png" alt="realtime" className="h-5 w-5 rounded-md border border-white/70 bg-white/90 p-0.5 shadow-sm object-contain" />
              <span>Xu ly real-time</span>
            </div>
          </div>
        </section>

        <section className="flex items-center justify-center md:justify-end">
          <div className="w-full max-w-md rounded-3xl border border-white/45 bg-white/78 p-8 shadow-elevated backdrop-blur-xl md:p-9">
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
                className="mt-2 rounded-2xl bg-blue-600 py-3 text-sm font-semibold text-white shadow-card transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
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
