import { expect, test } from "bun:test";

import {
  captureDisclosureState,
  reconcileChildren,
  restoreDisclosureState,
} from "./dom.js";

function child(name) {
  return {
    name,
    parent: null,
    remove() {
      const index = this.parent?.children.indexOf(this) ?? -1;
      if (index >= 0) this.parent.children.splice(index, 1);
      this.parent = null;
    },
  };
}

function parentWith(...initialChildren) {
  const parent = {
    children: [...initialChildren],
    get lastElementChild() {
      return this.children.at(-1) || null;
    },
    insertBefore(node, before) {
      let targetIndex = before ? this.children.indexOf(before) : this.children.length;
      const previousIndex = this.children.indexOf(node);
      if (previousIndex >= 0) {
        this.children.splice(previousIndex, 1);
        if (previousIndex < targetIndex) targetIndex -= 1;
      }
      this.children.splice(targetIndex, 0, node);
      node.parent = this;
    },
  };
  for (const node of parent.children) node.parent = parent;
  return parent;
}

test("reconciles children without replacing nodes that are already in place", () => {
  const first = child("first");
  const second = child("second");
  const stale = child("stale");
  const added = child("added");
  const parent = parentWith(first, second, stale);

  reconcileChildren(parent, [first, added, second]);

  expect(parent.children).toEqual([first, added, second]);
  expect(first.parent).toBe(parent);
  expect(second.parent).toBe(parent);
  expect(stale.parent).toBeNull();
});

test("captures and restores both open and closed disclosure state", () => {
  const process = {
    dataset: { disclosureKey: "turn:process" },
    open: true,
  };
  const tool = {
    dataset: { disclosureKey: "turn:tool" },
    open: false,
  };
  const root = {
    querySelectorAll() {
      return [process, tool];
    },
  };

  const state = captureDisclosureState(root);
  process.open = false;
  tool.open = true;
  restoreDisclosureState(root, state);

  expect(process.open).toBe(true);
  expect(tool.open).toBe(false);
});
