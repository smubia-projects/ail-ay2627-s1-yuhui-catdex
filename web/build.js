// Vercel build step: writes config.js with the API's address (VITE_API_BASE, set on Vercel).
const fs = require("fs");
const apiBase = (process.env.VITE_API_BASE || "http://localhost:8000").replace(/\/+$/, "");
fs.writeFileSync(__dirname + "/config.js", `window.VITE_API_BASE = ${JSON.stringify(apiBase)};\n`);
console.log("config.js ->", apiBase);
