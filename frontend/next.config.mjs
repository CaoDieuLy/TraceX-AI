/** @type {import('next').NextConfig} */
const internalApiGatewayUrl =
  process.env.INTERNAL_API_GATEWAY_URL ??
  process.env.NEXT_PUBLIC_API_GATEWAY_URL ??
  "http://backend:8000";

const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/search",
        destination: `${internalApiGatewayUrl}/search`,
      },
      {
        source: "/videos/:path*",
        destination: `${internalApiGatewayUrl}/videos/:path*`,
      },
      {
        source: "/api/v1/:path*",
        destination: `${internalApiGatewayUrl}/api/v1/:path*`,
      },
    ];
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "picsum.photos",
        pathname: "/**",
      },
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
        pathname: "/**",
      },
      {
        protocol: "http",
        hostname: "127.0.0.1",
        port: "8000",
        pathname: "/**",
      },
    ],
  },
};

export default nextConfig;
