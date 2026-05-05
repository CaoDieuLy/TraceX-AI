/** @type {import('next').NextConfig} */
const internalApiGatewayUrl =
  process.env.INTERNAL_API_GATEWAY_URL ??
  process.env.NEXT_PUBLIC_API_GATEWAY_URL ??
  "http://metadata-service:8000";

const trackingServiceUrl = process.env.TRACKING_SERVICE_URL ?? "";
const lightningApiToken = process.env.LIGHTNING_API_TOKEN ?? "";

const nextConfig = {
  output: "standalone",
  async rewrites() {
    const rules = [
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
      // API user (tranh GET /users bi middleware redirect sang trang HTML /admin/users)
      {
        source: "/api/users",
        destination: `${internalApiGatewayUrl}/users`,
      },
      {
        source: "/api/users/:path*",
        destination: `${internalApiGatewayUrl}/users/:path*`,
      },
      {
        source: "/auth/:path*",
        destination: `${internalApiGatewayUrl}/auth/:path*`,
      },
      {
        source: "/users",
        destination: `${internalApiGatewayUrl}/users`,
      },
      {
        source: "/users/:path*",
        destination: `${internalApiGatewayUrl}/users/:path*`,
      },
    ];

    // Proxy storage endpoints to LightningAI tracking service
    if (trackingServiceUrl) {
      rules.push(
        {
          source: "/api/storage/move",
          destination: `${trackingServiceUrl}/api/v1/storage/move`,
        },
        {
          source: "/api/storage/status",
          destination: `${trackingServiceUrl}/api/v1/storage/status`,
        }
      );
    }

    return rules;
  },
  images: {
    remotePatterns: [
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
      {
        protocol: "http",
        hostname: "localhost",
        port: "8001",
        pathname: "/**",
      },
      {
        protocol: "http",
        hostname: "127.0.0.1",
        port: "8001",
        pathname: "/**",
      },
    ],
  },
};

export default nextConfig;
