"use strict";

const pptxgen = require("pptxgenjs");
const deck = new pptxgen();

if (!deck || typeof deck.writeFile !== "function") {
  throw new Error("PptxGenJS runtime failed its self-check");
}

console.log(`PptxGenJS ${deck.version || "available"}`);
