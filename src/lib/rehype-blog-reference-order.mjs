function elementName(node) {
  return node?.type === "element" ? node.tagName : "";
}

function hasClass(node, className) {
  const classValue = node?.properties?.className;
  return Array.isArray(classValue)
    ? classValue.includes(className)
    : String(classValue || "")
        .split(/\s+/)
        .includes(className);
}

function isReferencesHeading(node) {
  return elementName(node) === "h2" && node.properties?.id === "references";
}

function isReferenceList(node) {
  return elementName(node) === "div" && hasClass(node, "reference-list");
}

function isFootnotes(node) {
  return elementName(node) === "section" && hasClass(node, "footnotes");
}

function isBlank(node) {
  return node?.type === "text" && String(node.value || "").trim() === "";
}

function nextContentIndex(children, start) {
  for (let index = start; index < children.length; index += 1) {
    if (!isBlank(children[index])) return index;
  }

  return -1;
}

function visit(node, callback) {
  if (!node || !Array.isArray(node.children)) return;
  callback(node);
  for (const child of node.children) visit(child, callback);
}

export default function rehypeBlogReferenceOrder() {
  return (tree) => {
    visit(tree, (node) => {
      const children = node.children;
      const referencesIndex = children.findIndex(isReferencesHeading);
      const footnotesIndex = children.findIndex(isFootnotes);
      if (referencesIndex === -1 || footnotesIndex === -1) return;

      const referenceHeading = children[referencesIndex];
      const referenceListIndex = nextContentIndex(children, referencesIndex + 1);
      const referenceList = children[referenceListIndex];
      if (!isReferenceList(referenceList)) return;
      if (referencesIndex > footnotesIndex) return;

      const movedNodes = children.splice(referencesIndex, referenceListIndex - referencesIndex + 1);
      const nextFootnotesIndex = children.findIndex(isFootnotes);
      children.splice(nextFootnotesIndex + 1, 0, ...movedNodes);
    });
  };
}
