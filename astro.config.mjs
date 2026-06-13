import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import { unified } from "@astrojs/markdown-remark";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkTyporaMath from "./src/lib/remark-typora-math.mjs";
import remarkBlogFigures from "./src/lib/remark-blog-figures.mjs";
import remarkBlogReferences from "./src/lib/remark-blog-references.mjs";
import rehypeBlogReferenceOrder from "./src/lib/rehype-blog-reference-order.mjs";

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
        remarkBlogFigures,
        remarkBlogReferences,
      ],
      rehypePlugins: [
        [rehypeKatex, { strict: false, throwOnError: false }],
        rehypeBlogReferenceOrder,
      ],
    }),
    shikiConfig: {
      theme: "github-light",
      wrap: true,
    },
  },
});
