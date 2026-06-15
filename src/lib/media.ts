export type MediaCategory = "Podcast" | "SpaceX" | "Badminton";

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
  featured: boolean;
};

export type MediaTrack = {
  title: string;
  url: string;
  videoId: string;
  source: string;
};

export type MediaPlaylistGroup = {
  key: "Featured" | MediaCategory;
  label: string;
  tracks: MediaTrack[];
};

export const curatedMedia: MediaItem[] = [
  {
    number: 1,
    title: "Whynot TV: Danfei Xu",
    host: "YouTube",
    addedDate: "2026-06-13",
    description: "A Whynot TV conversation with Danfei Xu on robotics research, embodied intelligence, and building systems that learn from real interaction.",
    note: "Robotics, research taste, and long-horizon AI systems.",
    youtubeUrl: "https://www.youtube.com/watch?v=__P5yygfRRQ&t=774s",
    tags: ["selected", "robotics"],
    category: "Podcast",
    featured: true,
  },
  {
    number: 2,
    title: "Starship | First Integrated Flight Test",
    host: "YouTube",
    addedDate: "2026-06-13",
    description: "SpaceX's first fully integrated Starship flight test, capturing the early full-stack attempt to launch the Starship and Super Heavy system together.",
    note: "The first integrated Starship launch attempt.",
    youtubeUrl: "https://www.youtube.com/watch?v=_krgcofiM6M",
    tags: ["selected", "spacex"],
    category: "SpaceX",
    featured: false,
  },
  {
    number: 3,
    title: "Whynot TV: Jiayi Weng",
    host: "YouTube",
    addedDate: "2026-06-15",
    description:
      "A Whynot TV profile of Jiayi Weng, tracing his path from open source and tool-building to OpenAI, where his work spans reinforcement learning, post-training, infrastructure, and major model releases.",
    note: "OpenAI, post-training, infra, and open-source values.",
    youtubeUrl: "https://www.youtube.com/watch?v=I0DrcsDf3Os",
    tags: ["selected", "ai", "openai"],
    category: "Podcast",
    featured: false,
  },
  {
    number: 4,
    title: "Starship | Second Flight Test",
    host: "YouTube",
    addedDate: "2026-06-15",
    description: "Starship's second integrated flight test, showing SpaceX's rapid iteration after the first launch and the next step toward a reusable heavy-lift launch system.",
    note: "A sharper second Starship test flight milestone.",
    youtubeUrl: "https://www.youtube.com/watch?v=C3iHAgwIYtI",
    tags: ["selected", "spacex", "starship"],
    category: "SpaceX",
    featured: false,
  },
  {
    number: 5,
    title: "Starship | Third Flight Test",
    host: "YouTube",
    addedDate: "2026-06-15",
    description: "Starship's third integrated flight test, continuing SpaceX's rapid iteration toward orbital-class operations and reusable heavy-lift launch capability.",
    note: "A third Starship test flight milestone.",
    youtubeUrl: "https://www.youtube.com/watch?v=ApMrILhTulI",
    tags: ["selected", "spacex", "starship"],
    category: "SpaceX",
    featured: false,
  },
  {
    number: 6,
    title: "Starship | Fourth Flight Test",
    host: "YouTube",
    addedDate: "2026-06-15",
    description: "Starship's fourth integrated flight test, following SpaceX's continued iteration across launch, stage separation, entry, and recovery objectives.",
    note: "A fourth Starship test flight milestone.",
    youtubeUrl: "https://www.youtube.com/watch?v=j2BdNDTlWbo",
    tags: ["selected", "spacex", "starship"],
    category: "SpaceX",
    featured: false,
  },
  {
    number: 7,
    title: "Starship | Fifth Flight Test",
    host: "YouTube",
    addedDate: "2026-06-15",
    description: "Starship's fifth integrated flight test, another step in SpaceX's campaign to mature the vehicle through repeated full-system launches.",
    note: "A fifth Starship test flight milestone.",
    youtubeUrl: "https://www.youtube.com/watch?v=hI9HQfCAw64",
    tags: ["selected", "spacex", "starship"],
    category: "SpaceX",
    featured: false,
  },
  {
    number: 8,
    title: "Starship | Sixth Flight Test",
    host: "YouTube",
    addedDate: "2026-06-15",
    description: "Starship's sixth integrated flight test, extending the sequence of high-tempo development flights for the Starship and Super Heavy system.",
    note: "A sixth Starship test flight milestone.",
    youtubeUrl: "https://www.youtube.com/watch?v=CMGiNKcVSek",
    tags: ["selected", "spacex", "starship"],
    category: "SpaceX",
    featured: false,
  },
  {
    number: 9,
    title: "Starship | Seventh Flight Test",
    host: "YouTube",
    addedDate: "2026-06-15",
    description: "Starship's seventh integrated flight test, documenting the next launch in SpaceX's iterative Starship development program.",
    note: "A seventh Starship test flight milestone.",
    youtubeUrl: "https://www.youtube.com/watch?v=Pn6e1O5bEyA",
    tags: ["selected", "spacex", "starship"],
    category: "SpaceX",
    featured: false,
  },
  {
    number: 10,
    title: "Starship | Tenth Flight Test",
    host: "YouTube",
    addedDate: "2026-06-15",
    description: "Starship's tenth integrated flight test, part of SpaceX's continuing sequence of Starship launch and recovery demonstrations.",
    note: "A tenth Starship test flight milestone.",
    youtubeUrl: "https://www.youtube.com/watch?v=rcd_SQZDlnk&list=PLBQ5P5txVQr9_jeZLGa0n5EIYvsOJFAnY&index=1",
    tags: ["selected", "spacex", "starship"],
    category: "SpaceX",
    featured: false,
  },
  {
    number: 11,
    title: "SpaceX | Starship - Test Like You Fly",
    host: "YouTube",
    addedDate: "2026-06-15",
    description: "A SpaceX Starship video focused on flight-like testing, the development philosophy behind rapid iteration, and preparing hardware for real mission conditions.",
    note: "Starship testing philosophy and development cadence.",
    youtubeUrl: "https://www.youtube.com/watch?v=ANe_HW4X8oc",
    tags: ["selected", "spacex", "starship"],
    category: "SpaceX",
    featured: true,
  },
  {
    number: 12,
    title: "Lin Dan | 2017 Retirement Interview",
    host: "YouTube",
    addedDate: "2026-06-16",
    description:
      "Lin Dan reflects beyond gold medals, thanking the great rival whose pressure helped shape the champion he became.",
    note: "Badminton, rivalry, and the meaning beyond gold medals.",
    youtubeUrl: "https://www.youtube.com/watch?v=tiTEG6gPhfU",
    tags: ["selected", "badminton", "podcast", "lin-dan"],
    category: "Badminton",
    featured: true,
  },
  {
    number: 13,
    title: "The Rise of SpaceX: Everything for Mars",
    host: "YouTube",
    addedDate: "2026-06-16",
    description:
      "A look at SpaceX's Mars-driven rise, from Falcon 1 failures to Starbase and a new space infrastructure system.",
    note: "Mars, Starbase, reusable rockets, and SpaceX's infrastructure ambition.",
    youtubeUrl: "https://www.youtube.com/watch?v=x2meJPOn9ws",
    tags: ["selected", "spacex", "mars", "podcast"],
    category: "SpaceX",
    featured: true,
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

export const featuredMedia = curatedMedia.filter((episode) => episode.featured);

function toMediaTracks(items: MediaItem[]): MediaTrack[] {
  return items
    .map((episode) => ({
      title: episode.title,
      url: episode.youtubeUrl,
      videoId: youtubeId(episode.youtubeUrl),
      source: episode.category,
    }))
    .filter((episode) => episode.videoId);
}

export const mediaPlaylist = toMediaTracks(featuredMedia);

export const mediaPlaylistGroups: MediaPlaylistGroup[] = [
  {
    key: "Featured",
    label: "Featured",
    tracks: mediaPlaylist,
  },
  {
    key: "Podcast",
    label: "Podcast",
    tracks: toMediaTracks(
      curatedMedia.filter((episode) => episode.category === "Podcast" || episode.tags.includes("podcast")),
    ),
  },
  {
    key: "SpaceX",
    label: "SpaceX",
    tracks: toMediaTracks(curatedMedia.filter((episode) => episode.category === "SpaceX")),
  },
  {
    key: "Badminton",
    label: "Badminton",
    tracks: toMediaTracks(curatedMedia.filter((episode) => episode.category === "Badminton")),
  },
].filter((group) => group.tracks.length);
