import type { GalleryPhoto } from "./gallery";

export function photoDetails(photo?: GalleryPhoto): [string, string | undefined][] {
  return [
    ["Camera", photo?.camera], ["Lens", photo?.lens],
    ["Focal length", photo?.focalLength], ["Aperture", photo?.aperture],
    ["Shutter speed", photo?.shutterSpeed], ["ISO", photo?.iso],
    ["Exposure", photo?.exposureCompensation], ["Resolution", photo?.resolution],
    ["Dimensions", photo ? `${photo.originalWidth ?? photo.width} × ${photo.originalHeight ?? photo.height} px` : undefined],
    ["File size", photo?.fileSize],
  ];
}

export function photoDate(takenAt?: string) {
  return takenAt?.replace("T", " · ").replaceAll("-", ".") ?? "Date not recorded";
}

const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const monthLabel = (takenAt: string) => `${months[Number(takenAt.slice(5, 7)) - 1]} ${takenAt.slice(0, 4)}`;

/** "Jul 2024", or "Jul 2024 – Aug 2024" when a collection spans months. */
export function collectionSpan(photos: GalleryPhoto[]) {
  const dates = photos.flatMap((photo) => (photo.takenAt ? [photo.takenAt] : [])).sort();
  return dates.length ? [...new Set([monthLabel(dates[0]), monthLabel(dates[dates.length - 1])])].join(" – ") : "";
}
