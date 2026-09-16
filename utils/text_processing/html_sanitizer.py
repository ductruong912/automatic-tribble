"""
HTML sanitization and escaping utilities for preventing Cross-Site Scripting (XSS).

Provides:
- escape_text: strict escaping for untrusted user inputs rendered in HTML contexts.
- SafeHTMLSanitizer: whitelist-based HTML parser that removes dangerous tags (<script>, <iframe>, <style>, etc.),
  strips event handlers (onerror, onclick, etc.), and validates URL schemes for safe tags like <a>.
- sanitize_html: convenience function to sanitize raw HTML strings.
"""

import html
import re
import urllib.parse
from html.parser import HTMLParser
from typing import Final, Optional


class SafeHTMLSanitizer(HTMLParser):
    """Whitelist-based HTML sanitizer using Python's standard library HTMLParser.
    
    Safe tags include common text styling, headers, lists, links, and tables.
    Dangerous elements (scripts, styles, embeds, iframes) and all their contents are removed.
    Stray or unclosed tags are balanced automatically.
    """

    ALLOWED_TAGS: Final[set[str]] = {
        "div", "p", "span", "br", "hr",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li",
        "strong", "b", "em", "i", "u", "s", "del",
        "a", "code", "pre", "blockquote",
        "table", "thead", "tbody", "tfoot", "tr", "th", "td"
    }

    VOID_TAGS: Final[set[str]] = {"br", "hr"}

    DISALLOWED_CONTENT_TAGS: Final[set[str]] = {
        "script", "style", "iframe", "object", "embed",
        "noscript", "meta", "link", "template", "svg",
        "math", "applet", "button", "input", "form",
        "select", "textarea"
    }

    ALLOWED_TAG_ATTRS: Final[dict[str, set[str]]] = {
        "a": {"href", "title", "target", "rel"},
        "code": {"class"},
        "span": {"class"},
        "div": {"class"},
        "p": {"class"},
        "th": {"align", "colspan", "rowspan"},
        "td": {"align", "colspan", "rowspan"},
    }

    SAFE_CLASS_RE: Final[re.Pattern] = re.compile(r"^[a-zA-Z0-9_\-\s]+$")
    DIGIT_RE: Final[re.Pattern] = re.compile(r"^\d+$")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self.tag_stack: list[str] = []
        self.drop_depth: int = 0

    def _is_safe_url(self, url: str) -> bool:
        """Validates that a URL does not use dangerous schemes like javascript: or data:."""
        if not url:
            return False
        clean_url = url.strip()
        # Reject control characters and DEL
        if any(ord(c) < 32 or ord(c) == 127 for c in clean_url):
            return False
        try:
            parsed = urllib.parse.urlparse(clean_url)
            scheme = parsed.scheme.lower()
            if scheme in ("http", "https", "mailto", "tel"):
                return True
            # Allow relative/anchor URLs
            if not scheme and (clean_url.startswith("/") or clean_url.startswith("#") or clean_url.startswith("./")):
                return True
        except Exception:
            return False
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self.DISALLOWED_CONTENT_TAGS:
            self.drop_depth += 1
            return
        if self.drop_depth > 0:
            return

        if tag not in self.ALLOWED_TAGS:
            return

        allowed_attrs = self.ALLOWED_TAG_ATTRS.get(tag, set())
        safe_attrs: list[str] = []
        is_external_link = False

        for attr_name, attr_val in attrs:
            attr_name = attr_name.lower()
            if not attr_val:
                attr_val = ""

            # Disallow all event handlers and inline styles
            if attr_name.startswith("on") or attr_name == "style":
                continue
            if attr_name not in allowed_attrs:
                continue

            if tag == "a" and attr_name == "href":
                if not self._is_safe_url(attr_val):
                    continue
                clean_href = attr_val.strip()
                safe_attrs.append(f'href="{html.escape(clean_href, quote=True)}"')
                parsed = urllib.parse.urlparse(clean_href)
                if parsed.scheme.lower() in ("http", "https"):
                    is_external_link = True
            elif attr_name == "class":
                clean_cls = attr_val.strip()
                if self.SAFE_CLASS_RE.match(clean_cls):
                    safe_attrs.append(f'class="{html.escape(clean_cls, quote=True)}"')
            elif tag in ("th", "td") and attr_name in ("colspan", "rowspan"):
                clean_num = attr_val.strip()
                if self.DIGIT_RE.match(clean_num):
                    safe_attrs.append(f'{attr_name}="{clean_num}"')
            elif tag in ("th", "td") and attr_name == "align":
                clean_align = attr_val.strip().lower()
                if clean_align in ("left", "center", "right", "justify"):
                    safe_attrs.append(f'align="{clean_align}"')
            elif tag == "a" and attr_name == "title":
                safe_attrs.append(f'title="{html.escape(attr_val.strip(), quote=True)}"')

        # Automatically add security attributes to external links
        if tag == "a" and is_external_link:
            safe_attrs.append('target="_blank"')
            safe_attrs.append('rel="noopener noreferrer"')

        attrs_str = (" " + " ".join(safe_attrs)) if safe_attrs else ""
        if tag in self.VOID_TAGS:
            self.output.append(f"<{tag}{attrs_str} />")
        else:
            self.output.append(f"<{tag}{attrs_str}>")
            self.tag_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.DISALLOWED_CONTENT_TAGS:
            if self.drop_depth > 0:
                self.drop_depth -= 1
            return
        if self.drop_depth > 0:
            return

        if tag in self.VOID_TAGS:
            return

        # Pop from stack until the matching tag is found (closes open child tags properly)
        if tag in self.tag_stack:
            while self.tag_stack:
                opened = self.tag_stack.pop()
                self.output.append(f"</{opened}>")
                if opened == tag:
                    break

    def handle_data(self, data: str) -> None:
        if self.drop_depth > 0:
            return
        # Escape raw characters so data cannot be re-interpreted as markup
        self.output.append(html.escape(data, quote=False))

    def handle_comment(self, data: str) -> None:
        # Strip all comments
        pass

    def get_safe_html(self) -> str:
        # Close any tags left open
        while self.tag_stack:
            opened = self.tag_stack.pop()
            self.output.append(f"</{opened}>")
        return "".join(self.output)


def escape_text(text: Optional[str]) -> str:
    """Escapes all HTML special characters in plain text for safe insertion into HTML.
    
    Converts &, <, >, ", and ' to their respective HTML entities.
    """
    if not text:
        return ""
    return html.escape(str(text), quote=True)


def sanitize_html(raw_html: Optional[str]) -> str:
    """Sanitizes an HTML string using a strict tag and attribute whitelist.
    
    Removes harmful elements (scripts, styles, iframes), drops event handlers,
    sanitizes URLs, and balances tags.
    """
    if not raw_html:
        return ""
    sanitizer = SafeHTMLSanitizer()
    sanitizer.feed(raw_html)
    sanitizer.close()
    return sanitizer.get_safe_html()
