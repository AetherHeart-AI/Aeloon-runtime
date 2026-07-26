export function reconcileChildren(parent, desiredChildren) {
  const desired = [...desiredChildren];
  for (const [index, child] of desired.entries()) {
    const current = parent.children[index] || null;
    if (current !== child) parent.insertBefore(child, current);
  }
  while (parent.children.length > desired.length) {
    parent.lastElementChild.remove();
  }
}

export function captureDisclosureState(root) {
  return new Map(
    [...root.querySelectorAll("details[data-disclosure-key]")].map((details) => [
      details.dataset.disclosureKey,
      details.open,
    ]),
  );
}

export function restoreDisclosureState(root, disclosureState) {
  for (const details of root.querySelectorAll("details[data-disclosure-key]")) {
    if (disclosureState.has(details.dataset.disclosureKey)) {
      details.open = disclosureState.get(details.dataset.disclosureKey);
    }
  }
}
