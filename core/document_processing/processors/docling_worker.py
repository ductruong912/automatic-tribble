"""
Standalone Docling OCR worker to avoid importing the main FastAPI app in Windows spawn.
Usage (internal): python -m core.processing.docling_worker <input_path> <kwargs_json>
Prints markdown to stdout.
"""

import sys
import json
import os


def main():
    if len(sys.argv) < 2:
        print("", end="")
        return 0

    input_path = sys.argv[1]
    kwargs = {}
    if len(sys.argv) >= 3:
        try:
            kwargs = json.loads(sys.argv[2])
        except Exception:
            kwargs = {}

    # Debug: environment and backend availability
    try:
        import torch as _torch
        cuda_ok = bool(getattr(_torch, "cuda", None) and _torch.cuda.is_available())
        print(f"__DEBUG__ torch={getattr(_torch,'__version__','unknown')} cuda={cuda_ok}", file=sys.stderr)
    except Exception as _e:
        print(f"__DEBUG__ torch:ERROR:{_e}", file=sys.stderr)
    try:
        import easyocr as _easy
        print(f"__DEBUG__ easyocr={getattr(_easy,'__version__','unknown')}", file=sys.stderr)
    except Exception as _e:
        print(f"__DEBUG__ easyocr:ERROR:{_e}", file=sys.stderr)
    print(
        f"__DEBUG__ env: OCR_ENABLED={os.getenv('DOCLING_OCR_ENABLED')} LANGS={os.getenv('DOCLING_OCR_LANGS')} DPI={os.getenv('DOCLING_OCR_DPI')} GPU={os.getenv('DOCLING_OCR_GPU')}",
        file=sys.stderr,
    )

    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"__ERROR__ Docling import failed: {e}", file=sys.stderr)
        return 2

    try:
        converter = DocumentConverter(**kwargs) if kwargs else DocumentConverter()
        result = converter.convert(input_path)
        try:
            md = result.document.export_to_markdown()  # type: ignore[attr-defined]
        except Exception:
            md = getattr(result, "markdown", "") or getattr(result, "text", "")
        if not md or not md.strip():
            # Treat empty markdown as error so the caller can try another variant or report clearly
            print("__ERROR__ Empty markdown output from Docling converter", file=sys.stderr)
            return 4
        # Emit a short success debug
        print(f"__DEBUG__ md_len={len(md)}", file=sys.stderr)
        print(md, end="")
        return 0
    except Exception as e:
        print(f"__ERROR__ Docling conversion failed: {e}", file=sys.stderr)
        return 3
    finally:
        try:
            import torch  # type: ignore
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())


