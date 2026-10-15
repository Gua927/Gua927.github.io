import fishingByTheSea from "../../public/assets/gallery/by-the-sea/2024-07-03_0631_001.json";
import lighthouseAtDawn from "../../public/assets/gallery/by-the-sea/2024-07-03_0508_001.json";
import afterglowOnTheWayHome from "../../public/assets/gallery/singles/2025-04-06_1824_001.json";
import theNewCamera from "../../public/assets/gallery/summer-palace/2026-10-11_1450_001.json";
import mandarinDuckOnTheWater from "../../public/assets/gallery/summer-palace/2026-10-11_1706_001.json";
import mandarinDuckInFullPlumage from "../../public/assets/gallery/summer-palace/2026-10-11_1709_001.json";
import mandarinDucksOnTheShore from "../../public/assets/gallery/summer-palace/2026-10-11_1721_001.json";
import pavilionBridgeAtNightfall from "../../public/assets/gallery/summer-palace/2026-10-11_1810_001.json";
import pavilionBridgeThroughTheWillows from "../../public/assets/gallery/summer-palace/2026-10-11_1812_001.json";
import yufengPagodaAboveTheWillows from "../../public/assets/gallery/singles/2026-10-13_1654_001.json";
import eyeToTheViewfinder from "../../public/assets/gallery/singles/2026-10-13_1715_001.json";
import dancingInTheSquare from "../../public/assets/gallery/singles/2026-10-14_1201_001.json";
import settingOut from "../../public/assets/gallery/singles/2026-10-14_2036_001.json";

export type GalleryPhoto = {
  id: string;
  title: string;
  // Themes describe individual photographs, independently of their collection.
  // Use only names from galleryThemes.
  themes: string[];
  // Short lowercase subject words shown as #tags, e.g. ["sea", "sunrise"].
  tags?: string[];
  // ISO date/month, optionally with local time: YYYY-MM-DDTHH:mm.
  takenAt?: string;
  notes?: string;
  location?: string;
  camera?: string;
  lens?: string;
  focalLength?: string;
  aperture?: string;
  shutterSpeed?: string;
  iso?: string;
  exposureCompensation?: string;
  originalFilename?: string;
  format?: string;
  resolution?: string;
  originalWidth?: number;
  originalHeight?: number;
  fileSize?: string;
  src: string;
  // A smaller copy shown on the wall and as thumbnails; the viewer always opens src.
  thumbnail?: string;
  width: number;
  height: number;
  alt: string;
  caption: string;
  // Featured photographs span two columns on the Gallery wall.
  featured?: boolean;
};

// The fixed set of themes, in the order the Gallery filters show them.
export const galleryThemes = ["Portraits", "Architecture", "Scenery", "Culture", "Animals & Plants"];

export type GalleryCollection = {
  slug: string;
  title: string;
  subject: string;
  description: string;
  preview: boolean;
  photos: GalleryPhoto[];
};

// Photo metadata lives beside each image in a JSON file. Keep its ID stable
// when replacing the image. The first photograph is the collection cover.
export const galleryCollections: GalleryCollection[] = [
  {
    slug: "by-the-sea",
    title: "First Visit to Qinhuangdao",
    subject: "Qinhuangdao",
    description: "Morning light, a lighthouse, and life along the shore on my first visit to Qinhuangdao.",
    preview: false,
    photos: [fishingByTheSea, lighthouseAtDawn],
  },
  {
    slug: "summer-palace",
    title: "Day One with My A7M5",
    subject: "Summer Palace",
    description: "The day my A7M5 arrived: unboxed in the afternoon, then straight out to the Summer Palace for mandarin ducks and Willow Bridge lit at nightfall.",
    preview: false,
    photos: [
      theNewCamera,
      mandarinDuckOnTheWater,
      mandarinDuckInFullPlumage,
      mandarinDucksOnTheShore,
      pavilionBridgeAtNightfall,
      pavilionBridgeThroughTheWillows,
    ],
  },
];

// Photographs that belong to no collection; they appear only on the Gallery wall.
export const gallerySingles: GalleryPhoto[] = [
  afterglowOnTheWayHome,
  yufengPagodaAboveTheWillows,
  eyeToTheViewfinder,
  dancingInTheSquare,
  settingOut,
];

// Fail the build on a misspelled or ad-hoc theme rather than growing the filter list.
for (const photo of [...galleryCollections.flatMap((collection) => collection.photos), ...gallerySingles]) {
  for (const theme of photo.themes) {
    if (!galleryThemes.includes(theme)) {
      throw new Error(`Photograph "${photo.id}" uses unknown theme "${theme}". Use one of: ${galleryThemes.join(", ")}.`);
    }
  }
  for (const tag of photo.tags ?? []) {
    if (!/^[a-z]+(-[a-z]+)*$/.test(tag)) {
      throw new Error(`Photograph "${photo.id}" has tag "${tag}". Tags are lowercase words, e.g. "sea" or "light-trail".`);
    }
  }
}
