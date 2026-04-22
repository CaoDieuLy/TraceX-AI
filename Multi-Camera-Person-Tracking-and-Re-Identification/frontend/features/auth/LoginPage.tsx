"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

export function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage(null);
    router.push("/home");
  }

  function handleRegister() {
    setMessage("Đăng ký chỉ là demo UI — chuyển tới trang chính.");
    router.push("/home");
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

          <button
            type="submit"
            className="mt-2 rounded-2xl bg-accent py-3 text-sm font-semibold text-white shadow-card transition hover:bg-accent-hover"
          >
            Login
          </button>
          <button
            type="button"
            onClick={handleRegister}
            className="rounded-2xl border border-surface-muted bg-white py-3 text-sm font-semibold text-ink shadow-card transition hover:bg-surface"
          >
            Đăng ký
          </button>
        </form>
      </div>
    </div>
  );
}
