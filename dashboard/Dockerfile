FROM node:22-bookworm-slim AS build
WORKDIR /app
ENV OBSERVABLE_TELEMETRY_DISABLE=true
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM nginxinc/nginx-unprivileged:stable-alpine
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
