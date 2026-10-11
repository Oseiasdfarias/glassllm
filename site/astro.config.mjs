import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://oseiasdfarias.github.io",
  base: "/glassllm",
  output: "static",
  trailingSlash: "always",
  devToolbar: { enabled: false },
  vite: {
    assetsInclude: ["**/*.whl"],
  },
  i18n: {
    defaultLocale: "pt",
    locales: ["pt", "en"],
    routing: { prefixDefaultLocale: true },
  },
});
