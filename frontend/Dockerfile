# Frontend Dockerfile for Docker/Coolify deployment
FROM node:20

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts 2>&1

# Copy source
COPY . .

# Build with full debug
SHELL ["/bin/bash", "-c"]
RUN set -x && \
    echo "=== Node version ===" && node --version && \
    echo "=== NPM version ===" && npm --version && \
    echo "=== Env vars ===" && env | grep -i next && \
    echo "=== Starting build ===" && \
    npm run build 2>&1

# Production image
FROM node:20

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

RUN groupadd --system --gid 1001 nodejs && \
    useradd --system --uid 1001 --gid nodejs nextjs

COPY --from=0 /app/public ./public
COPY --from=0 --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=0 --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]
