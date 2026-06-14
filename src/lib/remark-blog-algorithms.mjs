import katex from "katex";

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderInlineMath(value) {
  let output = "";
  let index = 0;

  while (index < value.length) {
    const start = value.indexOf("$", index);
    if (start === -1) {
      output += escapeHtml(value.slice(index));
      break;
    }

    const end = value.indexOf("$", start + 1);
    if (end === -1) {
      output += escapeHtml(value.slice(index));
      break;
    }

    output += escapeHtml(value.slice(index, start));

    const math = value.slice(start + 1, end);
    try {
      output += katex.renderToString(math, {
        displayMode: false,
        output: "html",
        strict: false,
        throwOnError: false,
      });
    } catch {
      output += escapeHtml(value.slice(start, end + 1));
    }

    index = end + 1;
  }

  return output;
}

function renderKeywords(value) {
  if (value.startsWith("input:")) {
    return `<span class="alg-key">Input:</span>${renderInlineMath(value.slice(6))}`;
  }

  if (value === "end while") {
    return '<span class="alg-key">end while</span>';
  }

  if (value.startsWith("while ") && value.endsWith(" do")) {
    const middle = value.slice(6, -3);
    return `<span class="alg-key">while</span> ${renderInlineMath(middle)} <span class="alg-key">do</span>`;
  }

  return renderInlineMath(value);
}

function renderLine(line) {
  const indent = line.match(/^\s*/)?.[0].length ?? 0;
  const trimmed = line.trim();
  const commentStart = trimmed.indexOf(" // ");
  const code = commentStart === -1 ? trimmed : trimmed.slice(0, commentStart);
  const comment = commentStart === -1 ? "" : trimmed.slice(commentStart + 4);
  const classes = ["algorithm-line"];

  if (indent > 0) classes.push("alg-indent");

  return [
    `<li class="${classes.join(" ")}">`,
    `<span class="algorithm-code">${renderKeywords(code)}</span>`,
    comment
      ? `<span class="alg-comment">${renderInlineMath(comment)}</span>`
      : "",
    "</li>",
  ].join("");
}

function transformAlgorithms(node) {
  if (!node || !Array.isArray(node.children)) return;

  for (let index = 0; index < node.children.length; index += 1) {
    const child = node.children[index];

    if (child.type === "code" && child.lang === "algorithm") {
      const lines = child.value.split("\n").filter((line) => line.trim() !== "");
      let caption = "Algorithm";
      const body = [];

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("caption:")) {
          caption = trimmed.slice(8).trim();
        } else {
          body.push(line);
        }
      }

      node.children[index] = {
        type: "html",
        value: [
          `<div class="algorithm-box" role="group" aria-label="${escapeHtml(caption)}">`,
          `<div class="algorithm-caption"><span>Algorithm 1</span> ${escapeHtml(caption)}</div>`,
          '<ol class="algorithm-lines">',
          ...body.map(renderLine),
          "</ol>",
          "</div>",
        ].join(""),
      };
      continue;
    }

    transformAlgorithms(child);
  }
}

export default function remarkBlogAlgorithms() {
  return (tree) => transformAlgorithms(tree);
}
