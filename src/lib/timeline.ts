import { getCollection } from "astro:content";
import { curatedMedia, youtubeId } from "./media";
import { galleryCollections, gallerySingles } from "./gallery";
import { orderGallery } from "./gallery-order";

export type TimelineCategory = "publication" | "blog" | "media" | "gallery" | "site-update";

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
  gallery: "Gallery",
  "site-update": "Site Update",
};

export const timelineOrder: TimelineCategory[] = [
  "publication",
  "blog",
  "media",
  "gallery",
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

// A collection is one entry dated by its earliest photograph; photographs outside
// any collection share one entry per day. Undated photographs stay off the timeline.
function galleryEntries(): TimelineEntry[] {
  const collections = galleryCollections.flatMap((collection) => {
    const first = collection.photos.flatMap((photo) => (photo.takenAt ? [photo.takenAt] : [])).sort()[0];
    if (!first) return [];
    const count = collection.photos.length;
    return [{
      id: `gallery-${collection.slug}`,
      date: new Date(first),
      title: collection.title,
      category: "gallery" as const,
      href: `/gallery/${collection.slug}/`,
      description: `${count} ${count === 1 ? "photograph" : "photographs"} · ${collection.description}`,
    }];
  });
  const days = new Map<string, typeof gallerySingles>();
  for (const photo of orderGallery(gallerySingles, "asc")) {
    if (!photo.takenAt) continue;
    const day = photo.takenAt.slice(0, 10);
    days.set(day, [...(days.get(day) ?? []), photo]);
  }
  const singles = [...days].map(([day, photos]) => {
    const first = photos[0];
    // The wall runs newest first, so the day's latest photograph is where it begins.
    const latest = photos[photos.length - 1];
    return {
      id: photos.length === 1 ? `gallery-${first.id}` : `gallery-${day}`,
      date: new Date(first.takenAt!),
      title: photos.map((photo) => photo.title).join(" · "),
      category: "gallery" as const,
      href: `/gallery/#photo-${latest.id}`,
      description: photos.length === 1 ? first.notes : `${photos.length} photographs`,
    };
  });
  return [...collections, ...singles];
}

export async function getTimelineEntries() {
  const [manualEntries, blogPosts, publications] = await Promise.all([
    getCollection("timeline"),
    getCollection("blog"),
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

  return [...manual, ...blog, ...publication, ...media, ...galleryEntries()].sort(sortTimeline);
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
