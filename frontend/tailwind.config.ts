import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./features/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#f7f7f8",
          card: "#ffffff",
          muted: "#ececf1",
        },
        ink: {
          DEFAULT: "#0d0d0d",
          secondary: "#676767",
          subtle: "#9b9b9b",
        },
        accent: {
          DEFAULT: "#10a37f",
          hover: "#0d8f70",
        },
      },
      boxShadow: {
        card: "0 2px 12px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        elevated: "0 8px 30px rgba(0,0,0,0.08)",
      },
    },
  },
  plugins: [],
};

export default config;
