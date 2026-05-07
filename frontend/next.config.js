/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  env: {
    NEXT_PUBLIC_API_BASE_URL: "https://tracex-ai.smartnovi.tech/api/v1",
  },
};

module.exports = nextConfig;
