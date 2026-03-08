"""Pure-Python ProseMirror JSON → Yjs ydoc state conversion.

Uses pycrdt (Rust-backed Yjs bindings) to build a Y.Doc with an XmlFragment
that mirrors the ProseMirror document tree. The output is binary-compatible
with the JavaScript ``yjs`` library.

This eliminates the need for Node.js / npm at runtime.
"""

from __future__ import annotations

import base64
from typing import Any


def prosemirror_to_ydoc_state(
    pm_doc: dict[str, Any],
    existing_ydoc_state: str | None = None,
) -> str | None:
    """Convert ProseMirror JSON to a base64-encoded Yjs state update.

    Args:
        pm_doc: ProseMirror document JSON (``{"type": "doc", "content": [...]}``)
        existing_ydoc_state: If provided, the existing ydoc state (base64) to
            merge against. The function will load this state, clear the existing
            prosemirror fragment, then write new content — producing a CRDT
            update that replaces rather than duplicates content.

    Returns:
        Base64-encoded Yjs state update, or ``None`` on failure.
    """
    try:
        from pycrdt import Doc, XmlElement, XmlFragment, XmlText
    except ImportError:
        return None

    # Build the new content as a fresh Y.Doc
    new_doc: Any = Doc()
    new_frag: Any = XmlFragment()
    new_doc["prosemirror"] = new_frag

    children = pm_doc.get("content", [])
    _insert_children(new_frag, children, XmlElement, XmlText)

    if existing_ydoc_state:
        # Load existing state, clear it, then merge with new content.
        existing_bytes = base64.b64decode(existing_ydoc_state)

        output_doc: Any = Doc()
        output_doc["prosemirror"] = XmlFragment()
        output_doc.apply_update(existing_bytes)

        # Clear existing prosemirror fragment
        frag = output_doc["prosemirror"]
        _clear_fragment(frag)

        # Merge: create a third doc that combines cleared state + new content
        merged_doc: Any = Doc()
        merged_doc["prosemirror"] = XmlFragment()
        merged_doc.apply_update(output_doc.get_update())
        merged_doc.apply_update(new_doc.get_update())

        update = merged_doc.get_update()
    else:
        update = new_doc.get_update()

    return base64.b64encode(update).decode("ascii")


def _insert_children(
    parent: Any,  # XmlFragment | XmlElement
    nodes: list[dict[str, Any]],
    xml_element_cls: type[Any],
    xml_text_cls: type[Any],
) -> None:
    """Recursively convert ProseMirror nodes and append to a Yjs parent."""
    # Group consecutive text nodes into runs (matching JS y-prosemirror behaviour)
    i = 0
    while i < len(nodes):
        node = nodes[i]
        node_type = node.get("type", "")

        if node_type == "text":
            # Collect consecutive text nodes
            text_run: list[dict[str, Any]] = []
            while i < len(nodes) and nodes[i].get("type") == "text":
                text_run.append(nodes[i])
                i += 1
            _insert_text_run(parent, text_run, xml_text_cls)
        else:
            _insert_element(parent, node, xml_element_cls, xml_text_cls)
            i += 1


def _insert_text_run(
    parent: Any,
    text_nodes: list[dict[str, Any]],
    xml_text_cls: type[Any],
) -> None:
    """Merge consecutive ProseMirror text nodes into a single XmlText."""
    xml_text = xml_text_cls()
    parent.children.append(xml_text)

    offset = 0
    for node in text_nodes:
        text = node.get("text", "")
        if not text:
            continue

        # Convert marks to attribute dict
        attrs = _marks_to_attrs(node.get("marks"))
        if attrs:
            xml_text.insert(offset, text, attrs)
        else:
            xml_text.insert(offset, text)
        offset += len(text)


def _insert_element(
    parent: Any,
    node: dict[str, Any],
    xml_element_cls: type[Any],
    xml_text_cls: type[Any],
) -> None:
    """Convert a ProseMirror element node to an XmlElement and append."""
    node_type = node.get("type", "unknown")
    el = xml_element_cls(node_type)
    parent.children.append(el)

    # Set node attributes (skip null values and "ychange")
    for key, val in node.get("attrs", {}).items():
        if val is not None and key != "ychange":
            el.attributes[key] = val

    # Recurse into children
    children = node.get("content", [])
    if children:
        _insert_children(el, children, xml_element_cls, xml_text_cls)


def _marks_to_attrs(marks: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Convert ProseMirror marks to Yjs text formatting attributes.

    Non-overlapping marks (standard: bold, italic, code, etc.) use the mark
    type name as the attribute key and the mark's attrs dict as the value.
    """
    if not marks:
        return {}
    attrs: dict[str, Any] = {}
    for mark in marks:
        mark_name = mark.get("type", "")
        mark_attrs = mark.get("attrs") or {}
        attrs[mark_name] = mark_attrs
    return attrs


def _clear_fragment(frag: Any) -> None:
    """Delete all children from an XmlFragment."""
    length = len(frag.children)
    if length > 0:
        del frag.children[0:length]
