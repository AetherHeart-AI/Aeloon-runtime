import { afterEach, beforeEach, expect, test } from "bun:test";

import { appendMarkdown } from "./markdown.js";

let originalDocument;

class FakeNode {
  constructor(type, value = "") {
    this.type = type;
    this.value = value;
    this.childNodes = [];
    this.dataset = {};
  }

  append(...children) {
    this.childNodes.push(...children);
  }

  get textContent() {
    return this.type === "text"
      ? this.value
      : this.childNodes.map((child) => child.textContent).join("");
  }

  set textContent(value) {
    this.childNodes = [new FakeNode("text", value)];
  }
}

beforeEach(() => {
  originalDocument = globalThis.document;
  globalThis.document = {
    createDocumentFragment: () => new FakeNode("fragment"),
    createElement: (tag) => new FakeNode(tag),
    createTextNode: (text) => new FakeNode("text", text),
  };
});

afterEach(() => {
  globalThis.document = originalDocument;
});

test("appends rendered Markdown nodes instead of stringifying the fragment", () => {
  const target = new FakeNode("target");

  appendMarkdown(target, "Fixed with **evidence**.");

  expect(target.childNodes).toHaveLength(1);
  expect(target.childNodes[0].type).toBe("fragment");
  expect(target.childNodes[0].childNodes[0].type).toBe("p");
  expect(target.textContent).toBe("Fixed with evidence.");
  expect(target.textContent).not.toContain("[object DocumentFragment]");
});
