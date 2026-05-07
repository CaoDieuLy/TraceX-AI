/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  env: {
    NEXT_PUBLIC_API_BASE_URL: "/api/v1",
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://metadata-service:8002/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
