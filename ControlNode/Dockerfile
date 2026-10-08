FROM node:24-bookworm
WORKDIR /usr/src/app
COPY package*.json ./
RUN npm ci
COPY . .
EXPOSE 8081
ENTRYPOINT ["node", "app.js"]
