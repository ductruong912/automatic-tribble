#!/usr/bin/env python3
"""
excel_to_chat_jsonl.py

Usage:
  python qa_to_jsonl.py --input "VNU_Q&A.xlsx" --output "train_chat.jsonl"

If your Excel has additional columns to teach function-calling, name them (case-insensitive)
- call_name / function_name / tool_name
- call_arguments / function_arguments / arguments  (can be JSON object or JSON string or plain text)
- tool_response / function_result / tool_output  (string or JSON string)
- tools  (stringified JSON array describing the tools schema; optional)
"""
import argparse
import json
import logging
import sys
from pathlib import Path
import unicodedata
import random
from typing import Optional

import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Helpers for header normalization and matching
def normalize_header(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ASCII", "ignore").decode("ASCII")
    return s.lower().strip()

QUESTION_KEYWORDS = {"cau hoi", "cauhoi", "cau_hoi", "cau-hoi", "question", "q", "quest", "cauhoi?"}
ANSWER_KEYWORDS = {"tra loi", "traloi", "tra_loi", "tra-loi", "answer", "a", "ans", "tra-lai"}

CALL_NAME_KEYS = {"call_name","function_name","tool_name","call_function","function"}
CALL_ARGS_KEYS = {"call_arguments","function_arguments","arguments","call_args","args"}
TOOL_RESP_KEYS = {"tool_response","function_result","tool_output","tool_response_raw","tool_result"}
TOOLS_SCHEMA_KEYS = {"tools","tools_schema","functions","function_schemas"}

def find_columns(df: pd.DataFrame):
    normalized = {col: normalize_header(col) for col in df.columns}
    q_col = None
    a_col = None
    call_name_col = None
    call_args_col = None
    tool_resp_col = None
    tools_schema_col = None

    for col, norm in normalized.items():
        # tokenize
        tokens = set(norm.replace("-", " ").replace("_", " ").split())
        joined = " ".join(tokens)
        if tokens & QUESTION_KEYWORDS and q_col is None:
            q_col = col
        if tokens & ANSWER_KEYWORDS and a_col is None:
            a_col = col
        if any(k in norm for k in CALL_NAME_KEYS) and call_name_col is None:
            call_name_col = col
        if any(k in norm for k in CALL_ARGS_KEYS) and call_args_col is None:
            call_args_col = col
        if any(k in norm for k in TOOL_RESP_KEYS) and tool_resp_col is None:
            tool_resp_col = col
        if any(k in norm for k in TOOLS_SCHEMA_KEYS) and tools_schema_col is None:
            tools_schema_col = col

    # fallback if no header detected: assume first two columns are Q/A
    if q_col is None or a_col is None:
        if len(df.columns) >= 2:
            logging.warning("Could not auto-detect headers; falling back to first two columns as Q/A.")
            q_col = q_col or df.columns[0]
            a_col = a_col or df.columns[1]

    return {
        "q_col": q_col,
        "a_col": a_col,
        "call_name_col": call_name_col,
        "call_args_col": call_args_col,
        "tool_resp_col": tool_resp_col,
        "tools_schema_col": tools_schema_col
    }

def safe_strify_arguments(value) -> Optional[str]:
    """Ensure function.arguments is a JSON string."""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        # if looks like JSON object/array, keep as-is but ensure it's valid JSON string
        if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
            try:
                # validate parseable JSON
                parsed = json.loads(v)
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                # not valid JSON; treat as raw string but escape quotes
                return json.dumps(v, ensure_ascii=False)
        else:
            # simple string: put into JSON string form (no extra quotes needed later)
            return json.dumps(v, ensure_ascii=False)
    else:
        # dict/list -> convert to compact JSON string
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            # fallback to string
            return json.dumps(str(value), ensure_ascii=False)

def safe_tool_content(value) -> str:
    """Tool content should be a string; if object given, stringify compactly."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)

def build_messages(row, cols, system_msg: Optional[str], idx: int):
    q = row.get(cols["q_col"], None)
    a = row.get(cols["a_col"], None)

    q_text = "" if pd.isna(q) else str(q).strip()
    a_text = None if pd.isna(a) else str(a).strip()

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})

    # user message: always include the raw question (if present)
    if q_text:
        messages.append({"role": "user", "content": q_text})

    # detect function/tool call columns exist for this row
    call_name = None
    call_args_raw = None
    tool_resp_raw = None
    tools_schema_raw = None

    if cols["call_name_col"]:
        call_name = row.get(cols["call_name_col"], None)
        if pd.isna(call_name):
            call_name = None
        else:
            call_name = str(call_name).strip()

    if cols["call_args_col"]:
        call_args_raw = row.get(cols["call_args_col"], None)
        if pd.isna(call_args_raw):
            call_args_raw = None

    if cols["tool_resp_col"]:
        tool_resp_raw = row.get(cols["tool_resp_col"], None)
        if pd.isna(tool_resp_raw):
            tool_resp_raw = None

    if cols["tools_schema_col"]:
        tools_schema_raw = row.get(cols["tools_schema_col"], None)
        if pd.isna(tools_schema_raw):
            tools_schema_raw = None

    obj = {"messages": messages}

    # If there's a function call, insert an assistant message with tool_calls
    tool_calls = None
    tools_array = None
    include_parallel = False

    if call_name:
        # stringify / validate arguments
        args_str = safe_strify_arguments(call_args_raw) if call_args_raw is not None else json.dumps({}, ensure_ascii=False)
        # We need function.arguments to be a string — often they are stored as JSON string.
        # If args_str already includes surrounding quotes (because safe_strify_arguments returns json.dumps),
        # we want to pass the inner JSON string as the value. The examples expect a string containing JSON,
        # so we will remove outer quotes only if args_str is itself a JSON string of a primitive.
        # Simpler approach: ensure arguments is a JSON string _without_ additional surrounding quotes:
        # we'll keep args_str as a JSON-serialized string (e.g. '{"location":"San Francisco"}').
        tool_calls = [
            {
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": call_name,
                    "arguments": args_str if isinstance(args_str, str) else json.dumps(args_str, ensure_ascii=False)
                }
            }
        ]
        obj["messages"].append({"role": "assistant", "tool_calls": tool_calls})

        # if there is a tool response, include role "tool" message with name
        if tool_resp_raw is not None:
            tool_content = safe_tool_content(tool_resp_raw)
            # If tool_content looks like JSON, leave it as string
            obj["messages"].append({"role": "tool", "name": call_name, "content": tool_content})

        # if there is an assistant final answer, include it
        if a_text:
            obj["messages"].append({"role": "assistant", "content": a_text})

        # build the tools array (either from provided tools_schema_raw or minimal auto schema)
        if tools_schema_raw:
            # if the cell contains JSON-like text, try to parse; otherwise assume it is already JSON string
            if isinstance(tools_schema_raw, str):
                try:
                    parsed = json.loads(tools_schema_raw)
                    tools_array = parsed
                except Exception:
                    # can't parse — attempt to wrap as a single function schema for the call_name
                    tools_array = [
                        {"type": "function",
                         "function": {
                             "name": call_name,
                             "description": "",
                             "parameters": {"type": "object", "properties": {}, "required": []}
                         }}
                    ]
            else:
                tools_array = tools_schema_raw
        else:
            # create compact minimal tool schema for the observed function name
            tools_array = [
                {"type": "function",
                 "function": {
                     "name": call_name,
                     "description": "",
                     "parameters": {"type": "object", "properties": {}, "required": []}
                 }}
            ]

        include_parallel = False

    else:
        # No function call: plain assistant message if answer exists
        if a_text:
            obj["messages"].append({"role": "assistant", "content": a_text})

    # attach tools and parallel_tool_calls top-level only if we constructed tools_array
    if tools_array:
        obj["tools"] = tools_array
        obj["parallel_tool_calls"] = include_parallel

    return obj

def write_jsonl(path: Path, objs):
    with path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Convert Excel Q&A (optionally function-call rows) to chat-style JSONL for fine-tuning.")
    parser.add_argument("--input", "-i", required=True, help="Input Excel file (.xlsx)")
    parser.add_argument("--sheet", "-s", default=None, help="Sheet name or index (optional)")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument("--system", default="Bạn là VNU JS:ER Assistant — chatbot hỗ trợ học thuật cho Tạp chí Khoa học VNU: Nghiên cứu Giáo dục (JS:ER), Đại học Quốc gia Hà Nội.", help="Optional system message to include")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (0 to disable)")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        logging.error(f"Input file not found: {inp}")
        sys.exit(2)

    logging.info(f"Reading Excel file: {inp}")
    try:
        if args.sheet is not None:
            df = pd.read_excel(inp, sheet_name=args.sheet, engine="openpyxl")
            if isinstance(df, dict):
                # unexpected but handle defensively
                first_sheet = list(df.keys())[0]
                logging.warning(f"sheet argument returned multiple sheets; using first sheet: {first_sheet}")
                df = df[first_sheet]
        else:
            all_sheets = pd.read_excel(inp, sheet_name=None, engine="openpyxl")
            if isinstance(all_sheets, dict):
                sheet_names = list(all_sheets.keys())
                logging.info(f"Found sheets: {sheet_names}")
                first = sheet_names[0]
                logging.info(f"Using first sheet: {first}")
                df = all_sheets[first]
            else:
                df = all_sheets
    except Exception as e:
        logging.exception("Failed to read Excel file. Ensure openpyxl is installed and the file is a valid .xlsx")
        raise

    logging.info(f"Columns found: {list(df.columns)}")
    cols = find_columns(df)
    logging.info(f"Detected columns: {cols}")
    if cols["q_col"] is None or cols["a_col"] is None:
        logging.error("Could not detect question/answer columns. Make sure the sheet contains them.")
        sys.exit(3)

    # iterate rows and build JSONL objects
    objs = []
    seed = args.seed
    for idx, r in enumerate(df.to_dict(orient="records"), start=1):
        obj = build_messages(r, cols, args.system, idx)
        # skip empty message arrays
        if not obj.get("messages"):
            continue
        objs.append(obj)

    logging.info(f"Built {len(objs)} JSONL examples.")
    # optionally shuffle to avoid order biases (if seed > 0)
    if seed and len(objs) > 1:
        random.seed(seed)
        random.shuffle(objs)

    out = Path(args.output)
    write_jsonl(out, objs)
    logging.info(f"Wrote JSONL to: {out} ({len(objs)} lines)")

if __name__ == "__main__":
    main()
