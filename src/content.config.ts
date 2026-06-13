import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const timeline = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/timeline" }),
  schema: z.object({
    date: z.date(),
    title: z.string(),
    category: z.enum(["publication", "blog", "media", "project", "site-update"]),
    href: z.string().optional(),
    description: z.string().optional(),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/projects" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.date(),
    tags: z.array(z.string()).default([]),
    image: z.string().optional(),
    href: z.string().optional(),
    status: z.enum(["active", "paused", "completed"]).default("active"),
  }),
});

const blog = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/blog" }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    excerpt: z.string(),
    category: z.string().optional(),
    tags: z.array(z.string()).default([]),
    author: z.string().default("Runze Tian"),
    affiliation: z.string().default("GenSI Lab, THU-AIR"),
    draft: z.boolean().default(false),
  }),
});

const publications = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/publications" }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    authors: z.array(
      z.object({
        name: z.string(),
        highlight: z.boolean().default(false),
      }),
    ),
    venue: z.string(),
    status: z.string(),
    abstract: z.string(),
    image: z.string(),
    imageAlt: z.string().default(""),
    pdf: z.string().optional(),
    blog: z.string().optional(),
    code: z.string().optional(),
  }),
});

export const collections = { timeline, projects, blog, publications };
