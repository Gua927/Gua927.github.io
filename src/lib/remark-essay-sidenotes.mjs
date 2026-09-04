const SIDENOTE_MARKERS = ["[!sidenote]", "[!旁注]"];

function transformSidenotes(node) {
  if (!node || !Array.isArray(node.children)) return;

  for (const child of node.children) {
    if (child.type === "blockquote") {
      const firstParagraph = child.children?.[0];
      const firstText = firstParagraph?.children?.[0];
      const marker = SIDENOTE_MARKERS.find((value) =>
        firstText?.type === "text" ? firstText.value.startsWith(value) : false,
      );

      if (marker) {
        firstText.value = firstText.value.slice(marker.length).replace(/^\s+/, "");

        if (!firstText.value) firstParagraph.children.shift();
        if (!firstParagraph.children.length) child.children.shift();

        child.data = {
          ...(child.data || {}),
          hName: "aside",
          hProperties: {
            ...child.data?.hProperties,
            className: ["essay-sidenote"],
            ariaLabel: "旁注",
          },
        };
      }
    }

    transformSidenotes(child);
  }
}

export default function remarkEssaySidenotes() {
  return (tree) => transformSidenotes(tree);
}
