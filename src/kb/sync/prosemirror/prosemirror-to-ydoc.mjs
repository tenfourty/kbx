#!/usr/bin/env node
/**
 * prosemirror-to-ydoc.mjs
 *
 * Reads ProseMirror JSON from stdin, converts it to a Yjs document update,
 * and outputs the base64-encoded Yjs state to stdout.
 *
 * If --existing-state <base64> is provided, loads that state first, clears
 * the existing prosemirror fragment, then writes new content. This produces
 * a CRDT update that replaces (not merges with) existing content.
 *
 * Usage:
 *   echo '{"type":"doc","content":[...]}' | node prosemirror-to-ydoc.mjs
 *   echo '{"type":"doc","content":[...]}' | node prosemirror-to-ydoc.mjs --existing-state <b64>
 *
 * Setup:
 *   cd <this-directory> && npm install
 */

import { createRequire } from 'module';
const require = createRequire(import.meta.url);

const Y = require('yjs');
const { prosemirrorJSONToYDoc } = require('y-prosemirror');
const { Schema } = require('prosemirror-model');

// --- ProseMirror schema matching Granola's editor ---

const schema = new Schema({
  nodes: {
    doc: {
      content: 'block+',
    },
    paragraph: {
      attrs: { id: { default: null } },
      content: 'inline*',
      group: 'block',
      parseDOM: [{ tag: 'p' }],
      toDOM() { return ['p', 0]; },
    },
    heading: {
      attrs: { id: { default: null }, level: { default: 1 } },
      content: 'inline*',
      group: 'block',
      defining: true,
      parseDOM: [
        { tag: 'h1', attrs: { level: 1 } },
        { tag: 'h2', attrs: { level: 2 } },
        { tag: 'h3', attrs: { level: 3 } },
        { tag: 'h4', attrs: { level: 4 } },
        { tag: 'h5', attrs: { level: 5 } },
        { tag: 'h6', attrs: { level: 6 } },
      ],
      toDOM(node) { return ['h' + node.attrs.level, 0]; },
    },
    bulletList: {
      content: 'listItem+',
      group: 'block',
      parseDOM: [{ tag: 'ul' }],
      toDOM() { return ['ul', 0]; },
    },
    orderedList: {
      attrs: { order: { default: 1 } },
      content: 'listItem+',
      group: 'block',
      parseDOM: [{ tag: 'ol' }],
      toDOM(node) {
        return node.attrs.order === 1
          ? ['ol', 0]
          : ['ol', { start: node.attrs.order }, 0];
      },
    },
    listItem: {
      content: 'block+',
      parseDOM: [{ tag: 'li' }],
      toDOM() { return ['li', 0]; },
      defining: true,
    },
    blockquote: {
      content: 'block+',
      group: 'block',
      parseDOM: [{ tag: 'blockquote' }],
      toDOM() { return ['blockquote', 0]; },
    },
    codeBlock: {
      content: 'text*',
      marks: '',
      group: 'block',
      code: true,
      defining: true,
      parseDOM: [{ tag: 'pre', preserveWhitespace: 'full' }],
      toDOM() { return ['pre', ['code', 0]]; },
    },
    horizontalRule: {
      group: 'block',
      parseDOM: [{ tag: 'hr' }],
      toDOM() { return ['hr']; },
    },
    text: {
      group: 'inline',
    },
  },
  marks: {
    bold: {
      parseDOM: [
        { tag: 'strong' },
        { tag: 'b' },
        { style: 'font-weight=bold' },
      ],
      toDOM() { return ['strong', 0]; },
    },
    italic: {
      parseDOM: [
        { tag: 'em' },
        { tag: 'i' },
        { style: 'font-style=italic' },
      ],
      toDOM() { return ['em', 0]; },
    },
    code: {
      parseDOM: [{ tag: 'code' }],
      toDOM() { return ['code', 0]; },
    },
  },
});

// --- Main ---

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf-8');
}

function parseArgs() {
  const args = process.argv.slice(2);
  let existingState = null;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--existing-state' && i + 1 < args.length) {
      existingState = args[i + 1];
      i++;
    }
  }
  return { existingState };
}

async function main() {
  const { existingState } = parseArgs();

  const input = (await readStdin()).trim();
  if (!input) {
    process.stderr.write('Error: no input on stdin\n');
    process.exit(1);
  }

  let pmJson;
  try {
    pmJson = JSON.parse(input);
  } catch (err) {
    process.stderr.write(`Error: invalid JSON on stdin: ${err.message}\n`);
    process.exit(1);
  }

  // Build the new content as a separate Y.Doc first
  const newDoc = prosemirrorJSONToYDoc(schema, pmJson, 'prosemirror');

  let outputDoc;

  if (existingState) {
    // Load existing state into a doc, clear it, then apply new content.
    // This produces a CRDT update that includes delete + insert operations,
    // so Granola's merge replaces the old content instead of duplicating.
    outputDoc = new Y.Doc();
    const existingBuf = Buffer.from(existingState, 'base64');
    Y.applyUpdate(outputDoc, existingBuf);

    // Clear existing prosemirror fragment
    const fragment = outputDoc.getXmlFragment('prosemirror');
    fragment.delete(0, fragment.length);

    // Copy new content from newDoc into outputDoc
    const newFragment = newDoc.getXmlFragment('prosemirror');
    const newUpdate = Y.encodeStateAsUpdate(newDoc);
    // Apply the new doc's state — but we need to merge carefully.
    // Instead, re-create content directly in outputDoc's fragment.
    // The simplest approach: encode newDoc as update relative to empty,
    // then we just send outputDoc's full state (which has the deletes + we add new content).

    // Actually, let's use a transactional approach:
    // We already cleared the fragment. Now we need to populate it with new content.
    // The easiest way: use prosemirrorJSONToYDoc on outputDoc directly.
    // But prosemirrorJSONToYDoc creates a new doc. So we extract the XML nodes.

    // Better approach: just serialize the cleared doc + apply new content via update
    const newState = Y.encodeStateAsUpdate(newDoc);

    // Create a fresh doc that combines: existing history (with deletes) + new content
    const mergedDoc = new Y.Doc();
    Y.applyUpdate(mergedDoc, Y.encodeStateAsUpdate(outputDoc)); // has delete ops
    // Now apply newDoc's content as a separate client
    Y.applyUpdate(mergedDoc, newState);

    outputDoc = mergedDoc;
  } else {
    outputDoc = newDoc;
  }

  // Encode full state
  const update = Y.encodeStateAsUpdate(outputDoc);
  const b64 = Buffer.from(update).toString('base64');
  process.stdout.write(b64);
}

main().catch((err) => {
  process.stderr.write(`Error: ${err.message}\n${err.stack}\n`);
  process.exit(1);
});
