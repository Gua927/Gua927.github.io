import { getCollection } from "astro:content";
import type { CollectionEntry } from "astro:content";
import { curatedMedia, youtubeId } from "./media";

export type TimelineCategory = "publication" | "blog" | "media" | "project" | "site-update";

export interface TimelineEntry {
  id: string;
  date: Date;
  title: string;
  category: TimelineCategory;
  href: string;
  description?: string;
}

export const timelineLabels: Record<TimelineCategory, string> = {
  publication: "Publication",
  blog: "Blog",
  media: "Media",
  project: "Project",
  "site-update": "Site Update",
};

export const timelineOrder: TimelineCategory[] = [
  "publication",
  "blog",
  "media",
  "project",
  "site-update",
];

function sortTimeline(a: TimelineEntry, b: TimelineEntry) {
  const byDate = b.date.getTime() - a.date.getTime();
  if (byDate !== 0) return byDate;

  const byCategory = timelineOrder.indexOf(a.category) - timelineOrder.indexOf(b.category);
  if (byCategory !== 0) return byCategory;

  return a.title.localeCompare(b.title);
}

function parseMediaDate(value: string) {
  if (/^\d{4}$/.test(value)) {
    return new Date(Number(value), 0, 1);
  }

  const parsed = new Date(value);
  if (!Number.isNaN(parsed.getTime())) return parsed;

  return new Date();
}

async function getOptionalProjects(): Promise<CollectionEntry<"projects">[]> {
  try {
    return await getCollection("projects");
  } catch {
    return [];
  }
}

export async function getTimelineEntries() {
  const [manualEntries, blogPosts, projects, publications] = await Promise.all([
    getCollection("timeline"),
    getCollection("blog"),
    getOptionalProjects(),
    getCollection("publications"),
  ]);

  const manual: TimelineEntry[] = manualEntries.map((entry) => ({
    id: `timeline-${entry.id}`,
    date: entry.data.date,
    title: entry.data.title,
    category: entry.data.category,
    href: entry.data.href ?? `/timeline/#timeline-${entry.id}`,
    description: entry.data.description,
  }));

  const blog: TimelineEntry[] = blogPosts
    .filter((post) => !post.data.draft)
    .map((post) => ({
      id: `blog-${post.id}`,
      date: post.data.date,
      title: post.data.title,
      category: "blog",
      href: `/blog/${post.id}/`,
      description: post.data.excerpt,
    }));

  const project: TimelineEntry[] = projects.map((project) => ({
    id: `project-${project.id}`,
    date: project.data.date,
    title: project.data.title,
    category: "project",
    href: project.data.href ?? `/project/#${project.id}`,
    description: project.data.description,
  }));

  const publication: TimelineEntry[] = publications.map((paper) => ({
    id: `publication-${paper.id}`,
    date: paper.data.date,
    title: paper.data.title,
    category: "publication",
    href: `/publication/#${paper.id}`,
    description: paper.data.venue,
  }));

  const media: TimelineEntry[] = curatedMedia.map((item) => {
    const videoId = youtubeId(item.youtubeUrl) || String(item.number);
    return {
      id: `media-${videoId}`,
      date: parseMediaDate(item.addedDate),
      title: item.title,
      category: "media",
      href: `/media/#media-${videoId}`,
      description: item.description,
    };
  });

  return [...manual, ...blog, ...project, ...publication, ...media].sort(sortTimeline);
}

export function getTimelineFilters(entries: TimelineEntry[]) {
  const counts = new Map<TimelineCategory, number>();
  for (const entry of entries) {
    counts.set(entry.category, (counts.get(entry.category) || 0) + 1);
  }

  return timelineOrder.map((category) => ({
    type: category,
    label: timelineLabels[category],
    count: counts.get(category) ?? 0,
  }));
}

export function formatTimelineDate(date: Date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}.${m}.${d}`;
}
