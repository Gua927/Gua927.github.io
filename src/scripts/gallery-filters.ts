import { orderGallery } from "../lib/gallery-order";

let cleanupFilters: (() => void) | undefined;

function initGalleryFilters() {
  cleanupFilters?.();
  const page = document.querySelector<HTMLElement>("[data-gallery-page]");
  const wall = page?.querySelector<HTMLElement>(".photo-wall");
  if (!page || !wall) return;

  const tiles = Array.from(wall.querySelectorAll<HTMLElement>(".photo-tile"));
  const themeFilters = Array.from(page.querySelectorAll<HTMLButtonElement>("[data-theme]"));
  const tagFilters = Array.from(page.querySelectorAll<HTMLButtonElement>("[data-tag]"));
  const search = page.querySelector<HTMLInputElement>("[data-gallery-search]");
  const sort = page.querySelector<HTMLSelectElement>("[data-gallery-sort]");
  const count = page.querySelector<HTMLElement>("[data-gallery-count]");
  const empty = page.querySelector<HTMLElement>("[data-gallery-empty]");
  const abort = new AbortController();
  const options = { signal: abort.signal };
  let theme = "all";
  let tag = "";

  function applyFilters() {
    const terms = (search?.value ?? "").normalize("NFKC").toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    const direction = sort?.value === "asc" ? "asc" : "desc";
    const matching = tiles.filter((tile) => {
      const text = (tile.dataset.search ?? "").normalize("NFKC").toLocaleLowerCase();
      const themes: string[] = JSON.parse(tile.dataset.themes ?? "[]");
      const tags: string[] = JSON.parse(tile.dataset.tags ?? "[]");
      return (theme === "all" || themes.includes(theme))
        && (!tag || tags.includes(tag))
        && terms.every((term) => text.includes(term));
    });
    const visible = orderGallery(matching.map((tile) => ({ tile, takenAt: tile.dataset.takenAt })), direction)
      .map(({ tile }) => tile);
    themeFilters.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.theme === theme)));
    tagFilters.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.tag === tag)));

    // Re-append the original tiles to keep their anchors and event handlers.
    wall!.replaceChildren(...visible);
    if (count) count.textContent = `${visible.length} ${visible.length === 1 ? "photograph" : "photographs"}`;
    if (empty) empty.hidden = visible.length > 0;
  }

  function clearFilters() {
    theme = "all";
    tag = "";
    if (search) search.value = "";
    applyFilters();
  }

  function revealPhoto(id: string) {
    const tile = tiles.find((item) => item.id === id);
    if (!tile) return;
    if (!wall!.contains(tile)) clearFilters();
    tile.querySelector<HTMLAnchorElement>(".gallery-photo")?.focus({ preventScroll: true });
    tile.scrollIntoView({
      behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
      block: "center",
    });
  }

  themeFilters.forEach((button) => button.addEventListener("click", () => {
    theme = button.dataset.theme ?? "all";
    applyFilters();
  }, options));
  // A second click on the active tag clears it.
  tagFilters.forEach((button) => button.addEventListener("click", () => {
    tag = button.dataset.tag === tag ? "" : button.dataset.tag ?? "";
    applyFilters();
  }, options));
  search?.addEventListener("input", applyFilters, options);
  sort?.addEventListener("change", applyFilters, options);
  page.querySelectorAll("[data-gallery-reset]").forEach((button) => button.addEventListener("click", clearFilters, options));
  function revealHash() {
    try { revealPhoto(decodeURIComponent(location.hash.slice(1))); } catch { /* Ignore malformed fragments. */ }
  }
  window.addEventListener("hashchange", revealHash, options);
  applyFilters();
  if (location.hash) requestAnimationFrame(revealHash);

  cleanupFilters = () => {
    abort.abort();
    cleanupFilters = undefined;
  };
}

document.addEventListener("astro:page-load", initGalleryFilters);
document.addEventListener("astro:before-swap", () => cleanupFilters?.());
initGalleryFilters();
