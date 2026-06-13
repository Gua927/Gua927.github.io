function isImageParagraph(node) {
  return (
    node?.type === "paragraph" &&
    Array.isArray(node.children) &&
    node.children.length === 1 &&
    node.children[0]?.type === "image"
  );
}

function visitChildren(node, callback) {
  if (!node || !Array.isArray(node.children)) return;
  callback(node);
  for (const child of node.children) visitChildren(child, callback);
}

export default function remarkBlogFigures() {
  return (tree) => {
    visitChildren(tree, (node) => {
      if (!Array.isArray(node.children)) return;

      node.children = node.children.map((child) => {
        if (!isImageParagraph(child)) return child;

        const image = child.children[0];
        if (!image.title) return child;

        const caption = image.title;
        image.title = null;

        return {
          type: "container",
          data: {
            hName: "figure",
            hProperties: { className: ["blog-figure"] },
          },
          children: [
            image,
            {
              type: "container",
              data: { hName: "figcaption" },
              children: [{ type: "text", value: caption }],
            },
          ],
        };
      });
    });
  };
}
