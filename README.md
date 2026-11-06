# Runze Tian Personal Site

Static personal homepage built with Astro.

## Commands

```sh
npm install
npm run dev
npm run build
```

The site is deployed from `dist/` by GitHub Actions.

## Photography

Store Gallery photographs in `public/assets/gallery/<collection>/`, with one
image and a matching JSON metadata file per photograph. Name both using the
capture time and a three-digit sequence: `YYYY-MM-DD_HHmm_NNN`.

```text
public/assets/gallery/by-the-sea/
  2024-07-03_0508_001.jpg
  2024-07-03_0508_001.json
  2024-07-03_0631_001.png
  2024-07-03_0631_001.json
```

The JSON file is the source of truth for the photo ID, display title, notes,
location, capture time, equipment, and file details. Import it in
`src/lib/gallery.ts` and add it to the relevant collection's `photos` array.
Photographs that belong to no collection live in `public/assets/gallery/singles/`
and are listed in `gallerySingles`; they appear only on the wall.
The Timeline follows the Gallery automatically: each collection is one entry dated
by its earliest photograph, and photographs outside a collection share one entry
per capture day, based on `takenAt`. Undated photographs are left off the Timeline.
Keep IDs stable when replacing images; increment the sequence for different
photos taken in the same minute. Collection display names may change while their
slugs and storage paths remain stable. Use `Place name, City` for locations,
naming the spot first where it is known (`Willow Bridge, Summer Palace, Beijing`).
The wall is a three-column masonry layout that keeps each photograph's own
proportions; set `"featured": true` in a photograph's JSON to let it span two
columns. The viewer always displays the whole image, scaled to fit the screen.
Keep camera originals offline (GitHub Pages sites are limited to 1 GB): publish a
display copy at most 4096 px on the long edge, JPEG quality 92, which looks the same
on any screen at a fraction of the size. For such large photographs, also add a
`thumbnail`: a 2000 px copy beside the image, named `YYYY-MM-DD_HHmm_NNN_thumb.jpg`,
which the wall, collection covers, and viewer thumbnails show instead of `src`.

Each photograph requires a stable, URL-safe `id`, `title`, `themes`, `src`, `width`, `height`, `alt`, and `caption`.
Optional details are `takenAt` (ISO date/month, optionally with local time), `notes`,
`location`, `camera`, `lens`, `focalLength`, `aperture`, `shutterSpeed`, `iso`,
`exposureCompensation`, `originalFilename`, `format`, `resolution`, `fileSize`, `tags`, `featured`, and `thumbnail`.
Use `originalWidth`/`originalHeight` for the original file's resolution while
`width`/`height` describe the displayed image asset; omit the original dimensions
when these are the same. `resolution` and `fileSize` also describe the original. Update dimensions, format, resolution, and file size
when replacing a file. `format` and `originalFilename` (the name of the uploaded file) are kept as
records; the viewer does not show them. Supply formatted
parameter values such as `50 mm`, `f/2.8`, `1/250 s`, and `100`. Omit unknown
values; the viewer shows them as —. Set `preview: true` only for
collections containing layout samples.

Themes filter the photo wall. Each photograph has its own `themes`, chosen from
the fixed list in `galleryThemes` (`src/lib/gallery.ts`): Portraits, Architecture,
Scenery, Culture, and Animals & Plants (for example, `["Culture", "Scenery"]`).
The build fails on any other theme name. Themes without photographs are shown
but disabled. `tags` are a few short lowercase subject words shown as `#tag`
(for example, `["sea", "sunrise"]`).
Reuse only the existing tags: `tower`, `sunrise`, `sea`, `sunset`, `afterglow`,
`heritage`, `autumn`, and `bird`. Do not introduce new tags when adding photographs.
If no existing tag fits, omit `tags` and keep the photograph's theme classification.
The Tags row lists them by frequency; click a tag to filter and again to
clear it. The search beside the title further narrows the results.
Date sorts newest first (default) or oldest first, with undated photographs last.
The wall fills columns in this order, as does the full-screen All viewer.
Collection links independently open `/gallery/<slug>/` directly in the full-screen
viewer, limited to that collection. Opening a photograph from the wall shows
the entire gallery, including photographs hidden by wall filters, with `Collection / All`.
Both entries share a large central photograph and a right sidebar with thumbnails
above the photograph's details. Use the thumbnails, previous/next buttons, the
edges of the photograph, swipes, or arrow keys to browse. Close or Escape returns
to the wall; closing a collection returns to `/gallery/`.

## Pinned News

The homepage News automatically shows only the three latest Publication and Blog
entries. Gallery updates remain on the Timeline. Manually specified announcements
can be added separately; a `PinnedNews` item appears before the automatic entries.
Configure its title, link, event date, and `from`/`until` timestamps in
`src/pages/index.astro`. Include an explicit timezone offset; `from` is inclusive
and `until` is exclusive. The browser shows the `pin` item only during this window,
including when returning through Astro navigation, and hides it at expiry without
requiring another build. The linked Gallery photograph and Timeline entry remain.
