import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import { unified } from "@astrojs/markdown-remark";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkTyporaMath from "./src/lib/remark-typora-math.mjs";

export default defineConfig({
  site: "https://Gua927.github.io",
  output: "static",
  integrations: [sitemap()],
  markdown: {
    processor: unified({
      remarkPlugins: [
        remarkGfm,
        remarkMath,
        remarkTyporaMath,
      ],
      rehypePlugins: [[rehypeKatex, { strict: false, throwOnError: false }]],
    }),
    shikiConfig: {
      theme: "github-light",
      wrap: true,
    },
  },
});
