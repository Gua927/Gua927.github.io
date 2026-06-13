# Blog Authoring Guide

This guide describes how to add or edit Blog posts for this Astro site. It is written for AI agents and contributors that need to create content matching the current Blog layout and behavior.

## Where Blog Posts Live

Create each post as a Markdown file under:

```text
src/content/blog/
```

The filename becomes the URL slug. For example:

```text
src/content/blog/sat-mask-diffusion-language-model-training.md
```

becomes:

```text
/blog/sat-mask-diffusion-language-model-training/
```

Use lowercase kebab-case filenames. Keep slugs stable after publication because links are derived from filenames.

## Required Frontmatter

Every Blog post must start with YAML frontmatter:

```yaml
---
title: "Post Title"
date: 2026-06-14
excerpt: A one-sentence summary shown in metadata and used as the page description.
category: notes
tags: ["TagA", "TagB"]
author: Runze Tian
affiliation: GenSI Lab, THU-AIR
draft: false
---
```

Field rules:

- `title` is required. It appears as the large page title and in the Blog index.
- `date` is required. Use `YYYY-MM-DD`.
- `excerpt` is required. Keep it concise, ideally one sentence.
- `category` is optional. If omitted, the site displays `notes`.
- `tags` is optional and defaults to `[]`. Use short labels; they appear as Blog filters.
- `author` is optional and defaults to `Runze Tian`.
- `affiliation` is optional and defaults to `GenSI Lab, THU-AIR`.
- `draft` is optional and defaults to `false`. Set `draft: true` to hide the post from the Blog index. Current route generation may still build draft pages, so do not use draft files for private content.

Known category order on the Blog index:

```text
pub-note, paper, notes, methods, ideas, essays, experiments
```

Other categories work, but they appear after the known categories.

## Metadata Links

On the Blog detail page:

- `author` is linked to the site home page `/`.
- `affiliation` is linked to `https://www.gensi-thuair.com/#/portal_home`.

If a future post needs different metadata links, update the Blog detail template rather than encoding links inside frontmatter.

## Markdown Structure

Write the body after frontmatter using regular Markdown.

Preferred heading hierarchy:

```markdown
# Optional top-level section

## Main section

### Subsection
```

The current design supports heading styles for:

- `#` / `h1`
- `##` / `h2`
- `###` / `h3`

Avoid `####` and deeper headings in Blog posts. They may render as browser/default Markdown headings but are not part of the designed Blog heading system or table of contents.

## Heading Visual Style

Blog body headings automatically receive a light gray `#` mark after the heading text:

- `h1` gets the largest `#` mark.
- `h2` gets a medium `#` mark.
- `h3` gets the smallest `#` mark.

Do not manually type decorative trailing `#` characters in headings. The CSS adds them automatically.

Good:

```markdown
## Train-inference mismatch
```

Avoid:

```markdown
## Train-inference mismatch #
```

## Table Of Contents

The right-side desktop table of contents is generated from Markdown headings with depth 1 through 3:

```text
h1, h2, h3
```

The TOC is hidden on mobile. On desktop it is sticky and highlights the currently active section while scrolling.

To keep the TOC usable:

- Use short, descriptive headings.
- Prefer 3 to 8 `h2` sections for normal notes.
- Use `h3` for real subsections, not for every paragraph.
- Avoid very long heading text. Long headings wrap and make the TOC tall.
- Do not skip levels, such as going directly from `##` to `####`.

## Content Length And Layout

The Blog detail page has:

- A large title and excerpt.
- A metadata row with Author, Affiliation, Published, and Category.
- Two full-width separator lines.
- A main content column.
- A sticky right-side TOC on desktop.

The content column is optimized for reading. Avoid manual HTML layout unless necessary.

Good Blog post shape:

```markdown
---
title: "Example Post"
date: 2026-06-14
excerpt: A short summary of the post.
category: notes
tags: ["Example"]
---

## Context

Introduce the problem and why it matters.

### Why this matters

Add a focused subsection only when it helps navigation.

## Method

Explain the approach.

## Results

Summarize the outcome.

## Takeaways

End with concise conclusions or open questions.
```

## Links

Use normal Markdown links:

```markdown
[GenSI Lab](https://www.gensi-thuair.com/#/portal_home)
```

External links are rendered like normal text links. Avoid raw bare URLs inside prose unless the URL itself is important.

## Footnotes

The Blog supports GFM footnotes. Use them for short explanatory notes that would interrupt the main paragraph.

```markdown
SAT-Mask is a schedule-level change.[^schedule]

[^schedule]: Here, "schedule" means the training-state construction rule, not a model architecture change.
```

Footnote behavior:

- The inline marker is rendered as a small superscript link.
- Footnotes are collected near the end of the rendered article.
- Footnote text is smaller and lighter than body text.
- Keep footnotes short. Use References for bibliographic citations instead of footnotes.

## Images

Blog-specific images should live next to the post's other assets:

```text
public/assets/blog/{post-slug}/fig/
```

`{post-slug}` must match the Markdown filename without `.md`. For example:

```text
src/content/blog/sat-mask-diffusion-language-model-training.md
public/assets/blog/sat-mask-diffusion-language-model-training/fig/
```

Reference them from Markdown with root-relative paths:

```markdown
![SAT-Mask method overview](/assets/blog/sat-mask-diffusion-language-model-training/fig/SAT-Mask.png)
```

Always provide meaningful alt text.

To add a visible caption, use the Markdown image title field:

```markdown
![SAT-Mask method overview](/assets/blog/sat-mask-diffusion-language-model-training/fig/SAT-Mask.png "Figure 1. Overview of SAT-Mask and its confidence-based partial denoising process.")
```

Caption behavior:

- The image is centered in the Blog body.
- The caption appears directly below the image.
- The caption box is centered with the image, but the caption text is left-aligned.
- Caption text is smaller than body text and uses a lighter color.
- Do not write separate HTML `<figure>` blocks unless a post needs custom layout.

## Math And Code

The site supports Markdown, GFM, and math rendering.

Inline math:

```markdown
Let $x_t$ be the masked state.
```

Display math:

```markdown
$$
p_\theta(x_0 \mid x_t)
$$
```

Code blocks should include a language when possible:

````markdown
```python
def schedule(t):
    return t
```
````

## References With BibTeX

Blog posts support BibTeX-backed references from standalone `.bib` files.

Use citation keys in the body with this syntax:

```markdown
Masked diffusion models use iterative denoising [@austin2021structured].
```

Multiple citations can be grouped:

```markdown
This line cites two papers [@austin2021structured; @sahoo2024simple].
```

Create a matching `.bib` file under:

```text
public/assets/blog/{post-slug}/references.bib
```

`{post-slug}` must match the Blog Markdown filename without `.md`.

For example:

```text
src/content/blog/sat-mask-diffusion-language-model-training.md
public/assets/blog/sat-mask-diffusion-language-model-training/references.bib
```

Recommended per-post asset layout:

```text
public/assets/blog/{post-slug}/
  references.bib
  fig/
    figure-1.png
    method-overview.svg
```

Put BibTeX entries in that file:

```bibtex
@article{austin2021structured,
  title = {Structured Denoising Diffusion Models in Discrete State-Spaces},
  author = {Austin, Jacob and Johnson, Daniel D. and Ho, Jonathan and Tarlow, Daniel and van den Berg, Rianne},
  journal = {Advances in Neural Information Processing Systems},
  year = {2021}
}
```

If a post needs a custom bibliography path, set `bib` in frontmatter:

```yaml
bib: public/assets/blog/custom-post/references.bib
```

The build pipeline will:

- convert `[@key]` citations into numbered inline links;
- append a `References` section at the end of the post;
- let readers hover or focus a citation link to preview the matching reference entry.

Reference rules:

- Citation keys in Markdown must match BibTeX entry keys in the `.bib` file.
- Use `;` between multiple citations.
- Keep BibTeX fields simple: `title`, `author`, `year`, `journal`, `booktitle`, `publisher`, `doi`, and `url` are supported.
- Do not put raw BibTeX blocks inside Blog Markdown; keep references in `public/assets/blog/{post-slug}/references.bib`.

## Tags And Filtering

Tags power the Blog index filters. Use stable, short tags. Existing examples:

```yaml
tags: ["MDLM", "Order"]
```

Prefer:

```yaml
tags: ["Diffusion", "Training"]
```

Avoid overly long tags:

```yaml
tags: ["A Very Long Research Theme Name"]
```

## Drafts

Use:

```yaml
draft: true
```

for work-in-progress posts that should not appear in the Blog index. Do not put sensitive or private content in draft posts unless route generation is also changed to exclude drafts.

## Checklist For Adding A Blog Post

Before finishing a new Blog post:

- The file is in `src/content/blog/`.
- The filename is lowercase kebab-case.
- Frontmatter includes `title`, `date`, and `excerpt`.
- `date` uses `YYYY-MM-DD`.
- `tags` is an array, even if empty.
- Headings use only `#`, `##`, and `###`.
- Headings are concise enough for the TOC.
- Blog images are stored in `public/assets/blog/{post-slug}/fig/`.
- Images use root-relative `/assets/blog/{post-slug}/fig/...` paths and alt text.
- References, if any, are stored in `public/assets/blog/{post-slug}/references.bib`.
- The post builds with `npm run build`.
