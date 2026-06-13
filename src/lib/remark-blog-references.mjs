import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const CITATION_PATTERN = /\[(@[^\]]+)\]/g;

function parseFrontmatter(value) {
  if (!value.startsWith("---")) return {};
  const end = value.indexOf("\n---", 3);
  if (end === -1) return {};

  const data = {};
  const raw = value.slice(3, end).split("\n");

  for (const line of raw) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;

    data[match[1]] = match[2]
      .trim()
      .replace(/^['"]|['"]$/g, "");
  }

  return data;
}

function bibFilename(title) {
  return title
    .replace(/[:"'`]/g, "")
    .replace(/[^A-Za-z0-9-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_+/g, "_");
}

function readPostFrontmatter(file) {
  const path = file?.path || file?.history?.[0];
  if (!path || !existsSync(path)) return {};

  return parseFrontmatter(readFileSync(path, "utf8"));
}

function externalBibPath(file) {
  const frontmatter = readPostFrontmatter(file);
  if (frontmatter.bib) {
    return resolve(process.cwd(), frontmatter.bib.replace(/^\//, ""));
  }

  if (!frontmatter.title) return "";

  const category = frontmatter.category || "notes";
  return resolve(
    process.cwd(),
    "bib",
    category,
    `${bibFilename(frontmatter.title)}.bib`,
  );
}

function normalizeValue(value) {
  return value
    .trim()
    .replace(/^\{|\}$/g, "")
    .replace(/^"|"$/g, "")
    .replace(/[{}]/g, "")
    .replace(/\\&/g, "&")
    .replace(/~/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function scanBibEntries(value) {
  const entries = [];
  let index = 0;

  while (index < value.length) {
    const at = value.indexOf("@", index);
    if (at === -1) break;

    const open = value.indexOf("{", at);
    if (open === -1) break;

    const type = value.slice(at + 1, open).trim().toLowerCase();
    let depth = 0;
    let close = -1;

    for (let i = open; i < value.length; i += 1) {
      const char = value[i];
      if (char === "{") depth += 1;
      if (char === "}") depth -= 1;
      if (depth === 0) {
        close = i;
        break;
      }
    }

    if (close === -1) break;

    const body = value.slice(open + 1, close);
    const comma = body.indexOf(",");
    if (comma > 0) {
      entries.push({
        type,
        key: body.slice(0, comma).trim(),
        fields: parseBibFields(body.slice(comma + 1)),
      });
    }

    index = close + 1;
  }

  return entries;
}

function parseBibFields(value) {
  const fields = {};
  let index = 0;

  while (index < value.length) {
    while (index < value.length && /[\s,]/.test(value[index])) index += 1;

    const nameStart = index;
    while (index < value.length && /[A-Za-z0-9_-]/.test(value[index])) index += 1;
    const name = value.slice(nameStart, index).toLowerCase();
    if (!name) break;

    while (index < value.length && /\s/.test(value[index])) index += 1;
    if (value[index] !== "=") break;
    index += 1;
    while (index < value.length && /\s/.test(value[index])) index += 1;

    let raw = "";
    if (value[index] === "{") {
      const start = index;
      let depth = 0;
      for (; index < value.length; index += 1) {
        if (value[index] === "{") depth += 1;
        if (value[index] === "}") depth -= 1;
        if (depth === 0) {
          index += 1;
          break;
        }
      }
      raw = value.slice(start, index);
    } else if (value[index] === '"') {
      const start = index;
      index += 1;
      while (index < value.length) {
        if (value[index] === '"' && value[index - 1] !== "\\") {
          index += 1;
          break;
        }
        index += 1;
      }
      raw = value.slice(start, index);
    } else {
      const start = index;
      while (index < value.length && value[index] !== ",") index += 1;
      raw = value.slice(start, index);
    }

    fields[name] = normalizeValue(raw);
  }

  return fields;
}

function formatAuthors(author) {
  if (!author) return "";
  return author
    .split(/\s+and\s+/i)
    .map((name) => name.trim())
    .filter(Boolean)
    .map((name) => {
      if (!name.includes(",")) return name;
      const [last, ...rest] = name.split(",").map((part) => part.trim());
      return [...rest, last].filter(Boolean).join(" ");
    })
    .join(", ");
}

function venue(entry) {
  return (
    entry.fields.journal ||
    entry.fields.booktitle ||
    entry.fields.publisher ||
    entry.fields.archiveprefix ||
    entry.type
  );
}

function referenceText(entry) {
  const authors = formatAuthors(entry.fields.author);
  const year = entry.fields.year ? `(${entry.fields.year}).` : "";
  const title = entry.fields.title ? `${entry.fields.title}.` : "";
  const where = venue(entry);
  const doi = entry.fields.doi ? `doi:${entry.fields.doi}` : "";
  const url = entry.fields.url || "";

  return [authors, year, title, where, doi, url].filter(Boolean).join(" ");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function citationKeys(value) {
  return value
    .split(/[;,]/)
    .map((part) => part.trim().replace(/^@/, ""))
    .filter(Boolean);
}

function citationGroups(value) {
  const groups = [];
  const pattern = /\[(@[^\]]+)\]/g;
  let cursor = 0;

  while (cursor < value.length) {
    pattern.lastIndex = cursor;
    const match = pattern.exec(value);
    if (!match) break;

    const keys = citationKeys(match[1]);
    const start = match.index;
    let end = pattern.lastIndex;

    while (end < value.length) {
      const nextStart = value.slice(end).match(/^\s*/)?.[0].length ?? 0;
      pattern.lastIndex = end + nextStart;
      const next = pattern.exec(value);
      if (!next || next.index !== end + nextStart) break;
      keys.push(...citationKeys(next[1]));
      end = pattern.lastIndex;
    }

    groups.push({ start, end, keys });
    cursor = end;
  }

  return groups;
}

function citationLabel(numbers) {
  const ranges = [];
  const sorted = [...numbers].sort((a, b) => a - b);

  for (let i = 0; i < sorted.length; i += 1) {
    const start = sorted[i];
    let end = start;

    while (i + 1 < sorted.length && sorted[i + 1] === end + 1) {
      i += 1;
      end = sorted[i];
    }

    if (end - start >= 2) {
      ranges.push(`${start}~${end}`);
    } else if (end > start) {
      ranges.push(String(start), String(end));
    } else {
      ranges.push(String(start));
    }
  }

  return `[${ranges.join(",")}]`;
}

function referenceMeta(entry) {
  const authors = formatAuthors(entry.fields.author);
  const year = entry.fields.year;
  const where = venue(entry);
  const doi = entry.fields.doi;

  return [where, authors, year, doi ? `doi:${doi}` : ""]
    .filter(Boolean)
    .join(" · ");
}

function citationPreview(entry, number) {
  if (!entry) {
    return `<span class="citation-preview-item"><span class="citation-preview-title">Missing reference</span></span>`;
  }

  const title = entry.fields.title || entry.key;
  const url = entry.fields.url;
  const pdf = url
    ? ` <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="citation-preview-pdf">[PDF]</a>`
    : "";

  return [
    `<span class="citation-preview-item">`,
    `<span class="citation-preview-index">[${number}]</span>`,
    `<span class="citation-preview-body">`,
    `<span class="citation-preview-title">${escapeHtml(title)}${pdf}</span>`,
    `<span class="citation-preview-meta">${escapeHtml(referenceMeta(entry))}</span>`,
    `</span>`,
    `</span>`,
  ].join("");
}

function visitChildren(node, callback) {
  if (!node || !Array.isArray(node.children)) return;
  callback(node);
  for (const child of node.children) visitChildren(child, callback);
}

function collectBibtex(tree, file) {
  const entries = new Map();
  const order = [];

  const path = externalBibPath(file);
  if (path && existsSync(path)) {
    for (const entry of scanBibEntries(readFileSync(path, "utf8"))) {
      if (!entry.key || entries.has(entry.key)) continue;
      entries.set(entry.key, entry);
      order.push(entry.key);
    }
  }

  visitChildren(tree, (node) => {
    if (!Array.isArray(node.children)) return;

    node.children = node.children.filter((child) => {
      if (
        child.type === "code" &&
        child.lang === "bibtex" &&
        typeof child.meta === "string" &&
        child.meta.split(/\s+/).includes("references")
      ) {
        for (const entry of scanBibEntries(child.value)) {
          if (!entry.key || entries.has(entry.key)) continue;
          entries.set(entry.key, entry);
          order.push(entry.key);
        }
        return false;
      }

      return true;
    });
  });

  return { entries, order };
}

function transformCitations(tree, entries, citationOrder) {
  const seen = new Set();

  function noteCitation(key) {
    if (!seen.has(key)) {
      seen.add(key);
      citationOrder.push(key);
    }
  }

  function transformNode(node) {
    if (!node || !Array.isArray(node.children) || ["link", "code", "inlineCode"].includes(node.type)) {
      return;
    }

    for (let i = 0; i < node.children.length; i += 1) {
      const child = node.children[i];

      const groups = child.type === "text" ? citationGroups(child.value) : [];
      if (groups.length > 0) {
        const parts = [];
        let lastIndex = 0;

        for (const group of groups) {
          if (group.start > lastIndex) {
            parts.push({ type: "text", value: child.value.slice(lastIndex, group.start) });
          }

          const keys = [...new Set(group.keys)];
          keys.forEach(noteCitation);
          const numbers = keys.map((key) => citationOrder.indexOf(key) + 1);
          const firstKey = keys[0];
          const firstEntry = entries.get(firstKey);
          const label = citationLabel(numbers);
          const title = keys
            .map((key) => {
              const entry = entries.get(key);
              return entry ? referenceText(entry) : `Missing reference: ${key}`;
            })
            .join("\n");
          const preview = keys.map((key) => {
            const number = citationOrder.indexOf(key) + 1;
            return citationPreview(entries.get(key), number);
          }).join("");
          const aria = firstEntry
            ? referenceText(firstEntry)
            : `Missing reference: ${firstKey}`;

          parts.push({
            type: "html",
            value: [
              `<span class="citation-shell">`,
              `<a href="#ref-${escapeHtml(firstKey)}" title="${escapeHtml(title)}" class="citation-link" aria-label="${escapeHtml(aria)}">`,
              `${escapeHtml(label)}`,
              `</a>`,
              `<span class="citation-preview">${preview}</span>`,
              `</span>`,
            ].join(""),
          });

          lastIndex = group.end;
        }

        if (lastIndex < child.value.length) {
          parts.push({ type: "text", value: child.value.slice(lastIndex) });
        }

        node.children.splice(i, 1, ...parts);
        i += parts.length - 1;
      } else {
        transformNode(child);
      }
    }
  }

  transformNode(tree);
}

function referenceItem(entry, index) {
  const authors = formatAuthors(entry.fields.author);
  const year = entry.fields.year;
  const title = entry.fields.title || entry.key;
  const where = venue(entry);
  const url = entry.fields.url;
  const doi = entry.fields.doi;

  const meta = [where, authors, year, doi ? `doi:${doi}` : ""]
    .filter(Boolean)
    .join(" · ");

  return {
    type: "container",
    data: {
      hProperties: {
        id: `ref-${entry.key}`,
        className: ["reference-item"],
        "data-reference-key": entry.key,
      },
    },
    children: [
      {
        type: "container",
        data: { hProperties: { className: ["reference-marker"] } },
        children: [{ type: "text", value: `[${index}]` }],
      },
      {
        type: "container",
        data: { hProperties: { className: ["reference-title"] } },
        children: [
          { type: "text", value: title },
          ...(url
            ? [
                { type: "text", value: " " },
                {
                  type: "link",
                  url,
                  data: { hProperties: { className: ["reference-pdf"] } },
                  children: [{ type: "text", value: "[PDF]" }],
                },
              ]
            : []),
        ],
      },
      {
        type: "container",
        data: { hProperties: { className: ["reference-meta"] } },
        children: [{ type: "text", value: meta }],
      },
    ],
  };
}

function appendReferences(tree, entries, bibOrder, citationOrder) {
  if (!entries.size) return;

  const orderedKeys = [
    ...citationOrder,
    ...bibOrder.filter((key) => !citationOrder.includes(key)),
  ].filter((key) => entries.has(key));

  if (!orderedKeys.length) return;

  tree.children.push({
    type: "heading",
    depth: 2,
    data: { hProperties: { id: "references" } },
    children: [{ type: "text", value: "References" }],
  });

  tree.children.push({
    type: "container",
    data: { hProperties: { className: ["reference-list"] } },
    children: orderedKeys.map((key, index) => referenceItem(entries.get(key), index + 1)),
  });
}

export default function remarkBlogReferences() {
  return (tree, file) => {
    const { entries, order: bibOrder } = collectBibtex(tree, file);
    if (!entries.size) return;

    const citationOrder = [];
    transformCitations(tree, entries, citationOrder);
    appendReferences(tree, entries, bibOrder, citationOrder);
  };
}
