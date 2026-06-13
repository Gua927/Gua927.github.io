export type PodcastEpisode = {
  number: number;
  title: string;
  host: string;
  date: string;
  description: string;
  note: string;
  youtubeUrl: string;
  tags: string[];
};

export type PodcastTrack = {
  title: string;
  url: string;
  videoId: string;
  source: string;
};

export const curatedPodcasts: PodcastEpisode[] = [
  {
    number: 1,
    title: "Whynot TV: Danfei Xu",
    host: "YouTube",
    date: "2026",
    description: "A selected Whynot TV conversation with Danfei Xu.",
    note: "A recommended long-form video for the Podcast shelf.",
    youtubeUrl: "https://www.youtube.com/watch?v=__P5yygfRRQ&t=774s",
    tags: ["selected", "robotics"],
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

export const podcastPlaylist: PodcastTrack[] = curatedPodcasts
  .map((episode) => ({
    title: episode.title,
    url: episode.youtubeUrl,
    videoId: youtubeId(episode.youtubeUrl),
    source: episode.host || "YouTube",
  }))
  .filter((episode) => episode.videoId);
