export function orderGallery<T extends { takenAt?: string }>(photos: T[], direction: "asc" | "desc" = "desc"): T[] {
  return [...photos].sort((a, b) => {
    // Unknown dates stay last in either direction; ties keep their archive order.
    if (!a.takenAt) return b.takenAt ? 1 : 0;
    if (!b.takenAt) return -1;
    const order = a.takenAt.localeCompare(b.takenAt);
    return direction === "asc" ? order : -order;
  });
}
