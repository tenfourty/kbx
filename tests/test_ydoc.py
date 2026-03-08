"""Tests for pure-Python ProseMirror → Yjs ydoc conversion."""

from __future__ import annotations

import base64

import pytest
from pycrdt import Doc, XmlElement, XmlFragment, XmlText

from kb.sync.ydoc import prosemirror_to_ydoc_state


# ---------------------------------------------------------------------------
# Fixtures: sample ProseMirror documents
# ---------------------------------------------------------------------------

SIMPLE_DOC = {
    "type": "doc",
    "content": [
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": "Hello world"}],
        }
    ],
}

HEADING_DOC = {
    "type": "doc",
    "content": [
        {
            "type": "heading",
            "attrs": {"level": 2, "id": None},
            "content": [{"type": "text", "text": "Title"}],
        },
        {
            "type": "paragraph",
            "content": [{"type": "text", "text": "Body text."}],
        },
    ],
}

MARKS_DOC = {
    "type": "doc",
    "content": [
        {
            "type": "paragraph",
            "content": [
                {"type": "text", "text": "Normal "},
                {
                    "type": "text",
                    "text": "bold",
                    "marks": [{"type": "bold"}],
                },
                {"type": "text", "text": " and "},
                {
                    "type": "text",
                    "text": "italic",
                    "marks": [{"type": "italic"}],
                },
                {"type": "text", "text": " text."},
            ],
        }
    ],
}

NESTED_LIST_DOC = {
    "type": "doc",
    "content": [
        {
            "type": "bulletList",
            "content": [
                {
                    "type": "listItem",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Item 1"}],
                        }
                    ],
                },
                {
                    "type": "listItem",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Item 2"}],
                        }
                    ],
                },
            ],
        }
    ],
}

COMPLEX_DOC = {
    "type": "doc",
    "content": [
        {
            "type": "heading",
            "attrs": {"level": 1, "id": "intro"},
            "content": [{"type": "text", "text": "Introduction"}],
        },
        {
            "type": "paragraph",
            "content": [
                {"type": "text", "text": "Some "},
                {
                    "type": "text",
                    "text": "bold and italic",
                    "marks": [{"type": "bold"}, {"type": "italic"}],
                },
                {"type": "text", "text": " text with "},
                {
                    "type": "text",
                    "text": "code",
                    "marks": [{"type": "code"}],
                },
                {"type": "text", "text": "."},
            ],
        },
        {
            "type": "blockquote",
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "A quote."}],
                }
            ],
        },
        {
            "type": "codeBlock",
            "content": [{"type": "text", "text": "print('hello')"}],
        },
    ],
}

EMPTY_DOC = {"type": "doc", "content": []}


# ---------------------------------------------------------------------------
# Helper: decode ydoc state and extract the prosemirror fragment
# ---------------------------------------------------------------------------


def _decode_ydoc(b64_state: str) -> Doc:
    """Decode a base64 ydoc state into a pycrdt Doc."""
    doc = Doc()
    doc["prosemirror"] = XmlFragment()
    doc.apply_update(base64.b64decode(b64_state))
    return doc


def _get_fragment_xml(doc: Doc) -> str:
    """Get the XML string representation of the prosemirror fragment."""
    return str(doc["prosemirror"])


def _count_children(frag: XmlFragment) -> int:
    """Count direct children of a fragment."""
    return len(frag.children)


# ---------------------------------------------------------------------------
# Tests: basic conversion
# ---------------------------------------------------------------------------


class TestBasicConversion:
    def test_simple_paragraph(self):
        result = prosemirror_to_ydoc_state(SIMPLE_DOC)
        assert result is not None
        doc = _decode_ydoc(result)
        frag = doc["prosemirror"]
        assert _count_children(frag) == 1
        child = frag.children[0]
        assert isinstance(child, XmlElement)
        assert child.tag == "paragraph"
        # Should have one XmlText child with "Hello world"
        assert len(child.children) == 1
        text_child = child.children[0]
        assert isinstance(text_child, XmlText)
        assert str(text_child) == "Hello world"

    def test_heading_with_attrs(self):
        result = prosemirror_to_ydoc_state(HEADING_DOC)
        assert result is not None
        doc = _decode_ydoc(result)
        frag = doc["prosemirror"]
        assert _count_children(frag) == 2

        heading = frag.children[0]
        assert isinstance(heading, XmlElement)
        assert heading.tag == "heading"
        assert heading.attributes["level"] == 2
        # id=None should be skipped
        assert "id" not in heading.attributes

        para = frag.children[1]
        assert isinstance(para, XmlElement)
        assert para.tag == "paragraph"

    def test_empty_doc(self):
        result = prosemirror_to_ydoc_state(EMPTY_DOC)
        assert result is not None
        doc = _decode_ydoc(result)
        frag = doc["prosemirror"]
        assert _count_children(frag) == 0


class TestMarks:
    def test_bold_and_italic_marks(self):
        result = prosemirror_to_ydoc_state(MARKS_DOC)
        assert result is not None
        doc = _decode_ydoc(result)
        frag = doc["prosemirror"]
        para = frag.children[0]
        assert isinstance(para, XmlElement)
        assert para.tag == "paragraph"

        # All consecutive text nodes should be merged into one XmlText
        assert len(para.children) == 1
        text = para.children[0]
        assert isinstance(text, XmlText)

        # Check the delta has the right formatting
        diff = text.diff()
        # diff is list of (content, attrs|None)
        texts = [chunk[0] for chunk in diff]
        assert "".join(texts) == "Normal bold and italic text."

        # Find the bold chunk
        bold_chunks = [c for c in diff if c[1] and "bold" in c[1]]
        assert len(bold_chunks) == 1
        assert bold_chunks[0][0] == "bold"

        # Find the italic chunk
        italic_chunks = [c for c in diff if c[1] and "italic" in c[1]]
        assert len(italic_chunks) == 1
        assert italic_chunks[0][0] == "italic"

    def test_multiple_marks_on_same_text(self):
        result = prosemirror_to_ydoc_state(COMPLEX_DOC)
        assert result is not None
        doc = _decode_ydoc(result)
        frag = doc["prosemirror"]

        # Second child is the paragraph with mixed marks
        para = frag.children[1]
        assert isinstance(para, XmlElement)
        text = para.children[0]
        assert isinstance(text, XmlText)

        diff = text.diff()
        # Find the "bold and italic" chunk — should have both marks
        bi_chunks = [
            c for c in diff if c[1] and "bold" in c[1] and "italic" in c[1]
        ]
        assert len(bi_chunks) == 1
        assert bi_chunks[0][0] == "bold and italic"

        # Find the "code" chunk
        code_chunks = [c for c in diff if c[1] and "code" in c[1]]
        assert len(code_chunks) == 1
        assert code_chunks[0][0] == "code"


class TestNestedStructure:
    def test_bullet_list(self):
        result = prosemirror_to_ydoc_state(NESTED_LIST_DOC)
        assert result is not None
        doc = _decode_ydoc(result)
        frag = doc["prosemirror"]

        # bulletList > listItem > paragraph > XmlText
        blist = frag.children[0]
        assert isinstance(blist, XmlElement)
        assert blist.tag == "bulletList"
        assert len(blist.children) == 2

        item1 = blist.children[0]
        assert isinstance(item1, XmlElement)
        assert item1.tag == "listItem"
        para1 = item1.children[0]
        assert isinstance(para1, XmlElement)
        assert para1.tag == "paragraph"
        text1 = para1.children[0]
        assert isinstance(text1, XmlText)
        assert str(text1) == "Item 1"

    def test_complex_doc_structure(self):
        result = prosemirror_to_ydoc_state(COMPLEX_DOC)
        assert result is not None
        doc = _decode_ydoc(result)
        frag = doc["prosemirror"]

        # heading, paragraph, blockquote, codeBlock
        assert _count_children(frag) == 4
        assert frag.children[0].tag == "heading"
        assert frag.children[1].tag == "paragraph"
        assert frag.children[2].tag == "blockquote"
        assert frag.children[3].tag == "codeBlock"

        # blockquote > paragraph > text
        bq = frag.children[2]
        assert len(bq.children) == 1
        bq_para = bq.children[0]
        assert isinstance(bq_para, XmlElement)
        assert bq_para.tag == "paragraph"


class TestMergeReplace:
    def test_merge_replaces_content(self):
        """Generate ydoc, then replace with new content via merge."""
        # First: create initial state
        initial_b64 = prosemirror_to_ydoc_state(SIMPLE_DOC)
        assert initial_b64 is not None

        # Second: replace with a different doc, passing existing state
        new_doc = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "Replaced content"}],
                }
            ],
        }
        merged_b64 = prosemirror_to_ydoc_state(new_doc, existing_ydoc_state=initial_b64)
        assert merged_b64 is not None

        # Verify the merged doc has the new content
        doc = _decode_ydoc(merged_b64)
        frag = doc["prosemirror"]
        # Should have new content
        para = frag.children[0]
        assert isinstance(para, XmlElement)
        text = para.children[0]
        assert isinstance(text, XmlText)
        assert str(text) == "Replaced content"

    def test_merge_without_existing_state(self):
        """Should work fine when no existing state is provided."""
        result = prosemirror_to_ydoc_state(SIMPLE_DOC, existing_ydoc_state=None)
        assert result is not None
        doc = _decode_ydoc(result)
        frag = doc["prosemirror"]
        assert _count_children(frag) == 1


class TestRoundTrip:
    def test_state_is_valid_base64(self):
        """Output should be valid base64 that decodes to bytes."""
        result = prosemirror_to_ydoc_state(SIMPLE_DOC)
        assert result is not None
        decoded = base64.b64decode(result)
        assert isinstance(decoded, bytes)
        assert len(decoded) > 0

    def test_state_can_be_applied_to_fresh_doc(self):
        """The output state should be applicable to a fresh Y.Doc."""
        result = prosemirror_to_ydoc_state(COMPLEX_DOC)
        assert result is not None

        # Apply to a completely fresh doc
        fresh_doc = Doc()
        fresh_doc["prosemirror"] = XmlFragment()
        fresh_doc.apply_update(base64.b64decode(result))

        frag = fresh_doc["prosemirror"]
        assert _count_children(frag) == 4

    def test_double_apply_is_idempotent(self):
        """Applying the same update twice should not duplicate content."""
        result = prosemirror_to_ydoc_state(SIMPLE_DOC)
        assert result is not None
        update = base64.b64decode(result)

        doc = Doc()
        doc["prosemirror"] = XmlFragment()
        doc.apply_update(update)
        doc.apply_update(update)  # second apply

        frag = doc["prosemirror"]
        # Should still be just one paragraph
        assert _count_children(frag) == 1
