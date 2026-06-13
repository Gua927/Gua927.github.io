export type MediaCategory = "Podcast" | "SpaceX";

export type MediaItem = {
  number: number;
  title: string;
  host: string;
  addedDate: string;
  description: string;
  note: string;
  youtubeUrl: string;
  tags: string[];
  category: MediaCategory;
};

export type MediaTrack = {
  title: string;
  url: string;
  videoId: string;
  source: string;
};

export const curatedMedia: MediaItem[] = [
  {
    number: 1,
    title: "Whynot TV: Danfei Xu",
    host: "YouTube",
    addedDate: "2026-06-13",
    description: "A selected Whynot TV conversation with Danfei Xu.",
    note: "A recommended long-form video for the Media shelf.",
    youtubeUrl: "https://www.youtube.com/watch?v=__P5yygfRRQ&t=774s",
    tags: ["selected", "robotics"],
    category: "Podcast",
  },
  {
    number: 2,
    title: "Starship | First Integrated Flight Test",
    host: "YouTube",
    addedDate: "2026-06-13",
    description: "A selected SpaceX video for the Media shelf.",
    note: "A SpaceX entry in the selected media shelf.",
    youtubeUrl: "https://www.youtube.com/watch?v=_krgcofiM6M",
    tags: ["selected", "spacex"],
    category: "SpaceX",
  },
];

export function youtubeId(url: string) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes("youtu.be")) {
      return parsed.pathname.split("/").filter(Boolean)[0] || "";
    }
    if (parsed.pathname.startsWith("/embed/")) {
      return parsed.pathname.split("/")[2] || "";
    }
    if (parsed.pathname.startsWith("/shorts/")) {
      return parsed.pathname.split("/")[2] || "";
    }
    return parsed.searchParams.get("v") || "";
  } catch {
    return "";
  }
}

export const mediaPlaylist: MediaTrack[] = curatedMedia
  .map((episode) => ({
    title: episode.title,
    url: episode.youtubeUrl,
    videoId: youtubeId(episode.youtubeUrl),
    source: episode.category,
  }))
  .filter((episode) => episode.videoId);
