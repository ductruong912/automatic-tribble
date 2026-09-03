from abc import ABC, abstractmethod
from typing import Optional
import io
import logging
import pandas as pd
from openpyxl import load_workbook

logger = logging.getLogger(__name__)


def smart_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Chunk free-form text on sentence boundaries.
    Delegates to TextUtils.chunk_text so plain text uses the same chunker as the rest
    of the pipeline (Vietnamese-aware sentence splitting).
    """
    from utils.text_processing.text_utils import TextUtils

    if not text or not text.strip():
        return []
    return TextUtils.chunk_text(text, chunk_size=chunk_size, overlap=overlap)


def smart_chunk_rows(rows: list, chunk_size: int = 1000, header: str = "") -> list:
    """
    Group tabular rows into chunks without ever splitting a row.
    The header line is repeated at the top of every chunk so each chunk stays
    self-describing when retrieved on its own.
    """
    rows = [r for r in rows if r and r.strip()]
    if not rows:
        return []

    prefix = f"{header}\n" if header else ""
    chunks = []
    current = []
    current_len = len(prefix)

    for row in rows:
        row_len = len(row) + 1  # +1 for the newline
        # Start a new chunk once this row would overflow, but never emit an empty one
        if current and current_len + row_len > chunk_size:
            chunks.append(prefix + "\n".join(current))
            current = []
            current_len = len(prefix)
        current.append(row)
        current_len += row_len

    if current:
        chunks.append(prefix + "\n".join(current))
    return chunks


class BaseFileExtractor(ABC):
    @abstractmethod
    async def extract(self, content: bytes, filename: Optional[str] = None) -> str:
        pass


class TXTTextExtractor(BaseFileExtractor):
    async def extract(
        self,
        content: bytes,
        filename: Optional[str] = None,
        chunk_size=1000,
        overlap=200,
    ) -> list:
        try:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = content.decode("latin-1")
                except UnicodeDecodeError:
                    text = content.decode("utf-8", errors="ignore")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            return smart_chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        except Exception:
            return []


class CSVTextExtractor(BaseFileExtractor):
    async def extract(
        self, content: bytes, filename: Optional[str] = None, chunk_size=1000
    ) -> list:
        try:
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    csv_text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                csv_text = content.decode("utf-8", errors="ignore")
            csv_file = io.StringIO(csv_text)
            df = pd.read_csv(csv_file)
            if df.empty:
                return []
            headers = " | ".join(df.columns)
            header = f"Columns: {headers}"

            # Process rows, handling NaN values properly
            rows = []
            for _, row in df.iterrows():
                # Convert each cell to string, handling NaN values
                row_cells = []
                for cell in row:
                    if pd.isna(cell):
                        row_cells.append("nan")
                    else:
                        row_cells.append(str(cell))
                row_str = " | ".join(row_cells)
                rows.append(row_str)

            return smart_chunk_rows(rows, chunk_size=chunk_size, header=header)
        except Exception as e:
            return []


class XLSXTextExtractor(BaseFileExtractor):
    async def extract(
        self, content: bytes, filename: Optional[str] = None, chunk_size=1000
    ) -> list:
        try:
            excel_file = io.BytesIO(content)
            # Use openpyxl with data_only=True so we get cached evaluated values instead of formulas
            wb = load_workbook(excel_file, data_only=True)
            all_chunks = []
            for ws in wb.worksheets:
                sheet_name = ws.title
                values = list(ws.iter_rows(values_only=True))
                if not values:
                    continue
                # Derive headers from first non-empty row; fallback to generic names
                header_row = None
                for r in values:
                    if any(cell is not None and str(cell).strip() != "" for cell in r):
                        header_row = r
                        break
                if header_row is None:
                    continue
                headers = [
                    (str(h).strip() if h is not None and str(h).strip() != "" else f"col_{i+1}")
                    for i, h in enumerate(header_row)
                ]
                header = f"Sheet: {sheet_name}\nColumns: {' | '.join(headers)}"

                # Build rows starting after the header row
                rows = []
                header_found = False
                for r in values:
                    if not header_found:
                        # skip until we've passed the header_row instance
                        if r is header_row:
                            header_found = True
                        continue
                    row_cells = []
                    empty_count = 0
                    for cell in r[: len(headers)]:
                        if cell is None:
                            row_cells.append("nan")
                            empty_count += 1
                        else:
                            # Format numbers to match Excel display (round to 2 decimal places)
                            if isinstance(cell, (int, float)):
                                if isinstance(cell, float) and cell.is_integer():
                                    row_cells.append(str(int(cell)))
                                else:
                                    row_cells.append(f"{cell:.2f}")
                            else:
                                row_cells.append(str(cell))
                    # Skip completely empty rows
                    if empty_count == len(headers):
                        rows.append("nan")  # marker to allow chunk splitter to break
                    else:
                        rows.append(" | ".join(row_cells))

                # Keep each sheet as one complete chunk instead of splitting by size
                # Filter out empty rows (marked as "nan")
                non_empty_rows = [row for row in rows if row != "nan"]
                
                if non_empty_rows:
                    # Create one chunk per sheet with all its data
                    sheet_content = "\n".join([header] + non_empty_rows)
                    all_chunks.append(sheet_content)
            return all_chunks
        except Exception as e:
            return []
