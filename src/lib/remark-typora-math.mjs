function splitInlineMath(value) {
  const parts = [];
  let rest = value;

  while (rest.length > 0) {
    const start = rest.indexOf("\\(");
    if (start === -1) {
      parts.push({ type: "text", value: rest });
      break;
    }

    const end = rest.indexOf("\\)", start + 2);
    if (end === -1) {
      parts.push({ type: "text", value: rest });
      break;
    }

    if (start > 0) {
      parts.push({ type: "text", value: rest.slice(0, start) });
    }

    parts.push({
      type: "inlineMath",
      value: rest.slice(start + 2, end),
    });
    rest = rest.slice(end + 2);
  }

  return parts.filter((part) => part.value !== "");
}

function visit(node) {
  if (!node || !Array.isArray(node.children)) return;

  for (let i = 0; i < node.children.length; i += 1) {
    const child = node.children[i];

    if (
      child.type === "paragraph" &&
      child.children?.length === 1 &&
      child.children[0].type === "text"
    ) {
      const value = child.children[0].value.trim();
      if (value.startsWith("\\[") && value.endsWith("\\]")) {
        node.children[i] = {
          type: "math",
          value: value.slice(2, -2).trim(),
        };
        continue;
      }
    }

    if (child.type === "text" && child.value.includes("\\(")) {
      node.children.splice(i, 1, ...splitInlineMath(child.value));
      i += splitInlineMath(child.value).length - 1;
      continue;
    }

    visit(child);
  }
}

export default function remarkTyporaMath() {
  return (tree) => visit(tree);
}
