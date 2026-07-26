export function parseMarkdown(source = "") {
  const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
  const nodes = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(/^```([\w+-]*)\s*$/);
    if (fence) {
      const content = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index])) {
        content.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      nodes.push({ type: "code", language: fence[1], text: content.join("\n") });
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      nodes.push({ type: "heading", level: heading[1].length, text: heading[2] });
      index += 1;
      continue;
    }

    const list = line.match(/^\s*(?:[-*+]|\d+\.)\s+(.+)$/);
    if (list) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      while (index < lines.length) {
        const match = lines[index].match(
          ordered ? /^\s*\d+\.\s+(.+)$/ : /^\s*[-*+]\s+(.+)$/,
        );
        if (!match) break;
        items.push(match[1]);
        index += 1;
      }
      nodes.push({ type: "list", ordered, items });
      continue;
    }

    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      const parts = [];
      while (index < lines.length) {
        const match = lines[index].match(/^>\s?(.*)$/);
        if (!match) break;
        parts.push(match[1]);
        index += 1;
      }
      nodes.push({ type: "quote", text: parts.join("\n") });
      continue;
    }

    const paragraph = [line.trim()];
    index += 1;
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(?:```|#{1,4}\s|>\s?|\s*(?:[-*+]|\d+\.)\s+)/.test(lines[index])
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    nodes.push({ type: "paragraph", text: paragraph.join(" ") });
  }
  return nodes;
}

export function renderMarkdown(source = "") {
  const fragment = document.createDocumentFragment();
  for (const node of parseMarkdown(source)) {
    let element;
    if (node.type === "code") {
      element = document.createElement("pre");
      const code = document.createElement("code");
      if (node.language) code.dataset.language = node.language;
      code.textContent = node.text;
      element.append(code);
    } else if (node.type === "heading") {
      element = document.createElement(`h${node.level}`);
      appendInline(element, node.text);
    } else if (node.type === "list") {
      element = document.createElement(node.ordered ? "ol" : "ul");
      for (const item of node.items) {
        const li = document.createElement("li");
        appendInline(li, item);
        element.append(li);
      }
    } else if (node.type === "quote") {
      element = document.createElement("blockquote");
      appendInline(element, node.text);
    } else {
      element = document.createElement("p");
      appendInline(element, node.text);
    }
    fragment.append(element);
  }
  return fragment;
}

function appendInline(parent, text) {
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\([^) \n]+\))/g;
  let cursor = 0;
  for (const match of String(text).matchAll(pattern)) {
    if (match.index > cursor) {
      parent.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    const token = match[0];
    if (token.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    } else if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else {
      const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      const anchor = document.createElement("a");
      anchor.textContent = link[1];
      if (/^https?:\/\//i.test(link[2])) {
        anchor.href = link[2];
        anchor.target = "_blank";
        anchor.rel = "noreferrer noopener";
      }
      parent.append(anchor);
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}
