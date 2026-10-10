import type { GalleryPhoto } from "../lib/gallery";
import { photoDate, photoDetails } from "../lib/gallery-details";

const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export class PhotoAlbumElement extends HTMLElement {
  private photos: GalleryPhoto[] = [];
  private currentIndex = 0;
  private events?: AbortController;
  // Only the most recent selection may swap the main image once it has decoded.
  private swapToken = 0;

  connectedCallback() {
    this.events?.abort();
    this.events = new AbortController();
    const options = { signal: this.events.signal };
    this.photos = JSON.parse(this.dataset.photos ?? "[]");
    this.addEventListener("click", (event) => {
      const target = event.target instanceof Element ? event.target : null;
      const thumb = target?.closest<HTMLElement>("[data-album-photo]");
      if (thumb) this.select(Number(thumb.dataset.albumPhoto));
      if (target?.closest("[data-album-prev]")) this.move(-1);
      if (target?.closest("[data-album-next]")) this.move(1);
    }, options);
    this.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        event.stopPropagation();
        this.move(event.key === "ArrowRight" ? 1 : -1);
      }
      if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        this.select(event.key === "Home" ? 0 : this.photos.length - 1);
      }
    }, options);

    // Horizontal swipes on the main photograph browse the album.
    const main = this.querySelector<HTMLElement>("[data-album-main]");
    let start: { x: number; y: number } | undefined;
    main?.addEventListener("pointerdown", (event) => {
      start = event.pointerType === "mouse" ? undefined : { x: event.clientX, y: event.clientY };
    }, options);
    main?.addEventListener("pointerup", (event) => {
      if (!start) return;
      const dx = event.clientX - start.x;
      const dy = event.clientY - start.y;
      start = undefined;
      if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy) * 1.5) this.move(dx < 0 ? 1 : -1);
    }, options);
    main?.addEventListener("pointercancel", () => { start = undefined; }, options);

    this.select(0, false);
  }

  disconnectedCallback() {
    this.events?.abort();
  }

  setPhotos(photos: GalleryPhoto[], index = 0, label = "Photographs") {
    this.photos = photos;
    this.dataset.photos = JSON.stringify(photos);
    this.dataset.label = label;
    const labelElement = this.querySelector("[data-album-label]");
    if (labelElement) labelElement.textContent = label;
    const thumbs = this.querySelector("[data-album-thumbs]");
    if (!thumbs) return;
    const fragment = document.createDocumentFragment();
    photos.forEach((photo, photoIndex) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "album-thumb";
      button.dataset.albumPhoto = String(photoIndex);
      button.setAttribute("aria-label", `Show photograph: ${photo.title}`);
      const image = document.createElement("img");
      image.src = photo.src;
      image.alt = "";
      image.width = photo.width;
      image.height = photo.height;
      image.decoding = "async";
      image.loading = "lazy";
      button.append(image);
      fragment.append(button);
    });
    thumbs.replaceChildren(fragment);
    this.select(index, false);
  }

  move(offset: number) {
    this.select(this.currentIndex + offset);
  }

  select(index: number, animate = true) {
    if (!this.photos.length) return;
    const nextIndex = Math.max(0, Math.min(index, this.photos.length - 1));
    const direction = Math.sign(nextIndex - this.currentIndex) || 1;
    const changed = nextIndex !== this.currentIndex;
    this.currentIndex = nextIndex;
    this.dataset.currentIndex = String(nextIndex);
    const photo = this.photos[nextIndex];

    this.showImage(photo, direction, animate && changed);
    this.updateThumbs();
    this.updateText(photo, animate && changed);
    this.querySelectorAll<HTMLButtonElement>("[data-album-prev]").forEach((button) => { button.disabled = nextIndex === 0; });
    this.querySelectorAll<HTMLButtonElement>("[data-album-next]").forEach((button) => { button.disabled = nextIndex === this.photos.length - 1; });

    // Warm the neighbours so the next step can cross-fade without waiting.
    for (const neighbour of [this.photos[nextIndex - 1], this.photos[nextIndex + 1]]) {
      if (neighbour) new Image().src = neighbour.src;
    }
  }

  private showImage(photo: GalleryPhoto, direction: number, animate: boolean) {
    const main = this.querySelector<HTMLElement>("[data-album-main]");
    const current = main?.querySelector<HTMLImageElement>("img.is-shown");
    if (!main || !current) return;
    const token = ++this.swapToken;

    if (!animate) {
      main.querySelectorAll("img:not(.is-shown)").forEach((image) => image.remove());
      Object.assign(current, { src: photo.src, alt: photo.alt, width: photo.width, height: photo.height });
      delete main.dataset.loading;
      return;
    }

    const incoming = current.cloneNode(false) as HTMLImageElement;
    incoming.className = "";
    incoming.removeAttribute("fetchpriority");
    Object.assign(incoming, { src: photo.src, alt: photo.alt, width: photo.width, height: photo.height });
    main.style.setProperty("--album-direction", String(direction));
    main.dataset.loading = "true";

    incoming.decode().catch(() => undefined).then(() => {
      if (token !== this.swapToken) return;
      delete main.dataset.loading;
      const outgoing = Array.from(main.querySelectorAll<HTMLImageElement>("img"));
      main.prepend(incoming);
      // Commit the starting position before moving to the shown state.
      incoming.getBoundingClientRect();
      incoming.classList.add("is-shown");
      outgoing.forEach((image) => {
        image.classList.remove("is-shown");
        image.classList.add("is-leaving");
        setTimeout(() => image.remove(), reducedMotion() ? 200 : 700);
      });
    });
  }

  private updateThumbs() {
    const container = this.querySelector<HTMLElement>("[data-album-thumbs]");
    this.querySelectorAll<HTMLButtonElement>("[data-album-photo]").forEach((thumb, thumbIndex) => {
      const active = thumbIndex === this.currentIndex;
      thumb.classList.toggle("is-current", active);
      thumb.setAttribute("aria-pressed", String(active));
      if (!active || !container) return;
      // Keep the current thumbnail visible without scrolling the page around it.
      const top = thumb.offsetTop - (container.clientHeight - thumb.offsetHeight) / 2;
      const left = thumb.offsetLeft - (container.clientWidth - thumb.offsetWidth) / 2;
      container.scrollTo({ top, left, behavior: reducedMotion() ? "instant" : "smooth" });
    });
  }

  private updateText(photo: GalleryPhoto, animate: boolean) {
    const text = (selector: string, value: string) => {
      const element = this.querySelector(selector);
      if (element) element.textContent = value;
    };
    text("[data-album-title]", photo.title);
    text("[data-album-date]", photoDate(photo.takenAt));
    text("[data-album-notes]", photo.notes || photo.caption || "No notes added.");
    text("[data-album-location]", photo.location || "—");
    text("[data-album-tags]", (photo.tags ?? []).map((tag) => `#${tag}`).join(" "));
    text("[data-album-position]", `${String(this.currentIndex + 1).padStart(2, "0")} / ${String(this.photos.length).padStart(2, "0")}`);
    const details = Object.fromEntries(photoDetails(photo));
    this.querySelectorAll<HTMLElement>("[data-album-field]").forEach((field) => {
      field.textContent = details[field.dataset.albumField!] || "—";
    });

    if (!animate || reducedMotion()) return;
    this.querySelectorAll<HTMLElement>("[data-album-fade]").forEach((element) => element.animate(
      [{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }],
      { duration: 360, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)" },
    ));
  }
}

if (!customElements.get("photo-album")) customElements.define("photo-album", PhotoAlbumElement);
