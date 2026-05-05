# Frontend Dockerfile for Docker/Coolify deployment
FROM node:20

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts

# Copy source
COPY . .

# Build with debug
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

RUN npm run build 2>&1; echo "EXIT_CODE=$?" >> /tmp/build_result.txt

# Check build result
RUN if [ -d .next ]; then echo "Build SUCCESS: .next exists"; ls -la .next/; else echo "Build FAILED: .next not found"; cat /tmp/build_result.txt; exit 1; fi

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
