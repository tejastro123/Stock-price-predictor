# Express Server Dockerfile
# =========================
FROM node:20-alpine

WORKDIR /app

# Install dependencies first (cache layer)
COPY package*.json ./
RUN npm ci --omit=dev

# Copy application code
COPY . .

EXPOSE 5000

CMD ["node", "server.js"]
