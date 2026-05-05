/** @type {import('next').NextConfig} */
const trackingServiceUrl = process.env.TRACKING_SERVICE_URL ?? "";
const lightningApiToken = process.env.LIGHTNING_API_TOKEN ?? "";

const nextConfig = {
  output: "standalone",
  async rewrites() {
    const rules = [
      {
        source: "/search",
        destination: `http://metadata-service:8002/search`,
      },
      {
        source: "/videos/:path*",
        destination: `http://metadata-service:8002/videos/:path*`,
      },
      {
        source: "/auth/:path*",
        destination: `http://metadata-service:8002/auth/:path*`,
      },
      {
        source: "/users",
        destination: `http://metadata-service:8002/users`,
      },
      {
        source: "/users/:path*",
        destination: `http://metadata-service:8002/users/:path*`,
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
