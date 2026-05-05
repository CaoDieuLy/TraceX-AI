/** @type {import('next').NextConfig} */
const trackingServiceUrl = process.env.TRACKING_SERVICE_URL ?? "";
const isCoolify = process.env.COOLIFY_DEPLOYMENT === "true";

// In Coolify, container name becomes the hostname
const metadataServiceHost = isCoolify ? "mcpt-metadata-service" : "metadata-service";
const metadataServicePort = "8002";

const nextConfig = {
  output: "standalone",
  async rewrites() {
    const base = `http://${metadataServiceHost}:${metadataServicePort}`;
    const rules = [
      {
        source: "/api/:path*",
        destination: `${base}/:path*`,
      },
      {
        source: "/search",
        destination: `${base}/search`,
      },
      {
        source: "/videos/:path*",
        destination: `${base}/videos/:path*`,
      },
      {
        source: "/auth/:path*",
        destination: `${base}/auth/:path*`,
      },
      {
        source: "/users",
        destination: `${base}/users`,
      },
      {
        source: "/users/:path*",
        destination: `${base}/users/:path*`,
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
        hostname: metadataServiceHost,
        port: metadataServicePort,
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
