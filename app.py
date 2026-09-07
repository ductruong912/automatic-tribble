import streamlit as st
import streamlit.components.v1 as components
import json
import os
import sys
import time
import subprocess
import requests
import re
import urllib3
import base64
from utils.file_operations import cleanup_data_folders


st.set_page_config(
    page_title="ChatBot",
    initial_sidebar_state="collapsed",
    layout="wide"
)
# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Logo
def get_image_base64(image_path):
    """Convert image to base64 for HTML embedding."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None
        
        
def is_backend_healthy(base_url: str) -> bool:
    try:
        # Disable SSL verification for self-signed certificates
        r = requests.get(f"{base_url}/", timeout=3, verify=False)
        return r.ok
    except Exception:
        return False


def stream_chat(base_url: str, query: str, history: list = None):
    """
    Read the Server-Sent Events stream from /chat/, yielding ("token", text) pairs and
    at most one ("error", message).

    Each frame carries a JSON payload, so line breaks inside the answer arrive intact.
    Splitting on "\\n" specifically - rather than letting requests use splitlines() -
    keeps rarer Unicode line separators inside a token from cutting a frame in half.
    """
    url = f"{base_url}/chat/"
    headers = {"accept": "text/event-stream", "content-type": "application/json"}
    payload = {"query": query}
    if history:
        payload["history"] = history
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=300, verify=False) as r:
        r.raise_for_status()

        data_lines = []
        for line in r.iter_lines(decode_unicode=True, delimiter="\n"):
            line = (line or "").rstrip("\r")

            if line:
                if line.startswith(":"):
                    continue  # Comment / keep-alive ping
                field, _, value = line.partition(":")
                if field == "data":
                    data_lines.append(value[1:] if value.startswith(" ") else value)
                continue

            # A blank line closes the frame
            if not data_lines:
                continue
            raw, data_lines = "\n".join(data_lines), []
            try:
                event = json.loads(raw)
            except ValueError:
                continue  # Ignore a frame we cannot parse rather than showing it as text

            if event.get("type") == "token":
                yield "token", event.get("text", "")
            elif event.get("type") == "error":
                yield "error", event.get("message") or "Unknown error"
                return
            elif event.get("type") == "done":
                return


def start_backend_subprocess(host: str, port: int, protocol: str = "http") -> bool:
    """Start uvicorn for the FastAPI app in a background subprocess."""
    try:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            "1",
        ]
        creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.Popen(cmd, creationflags=creation)
        # Wait briefly for server to come up
        for _ in range(30):
            time.sleep(0.2)
            if is_backend_healthy(f"{protocol}://{host}:{port}"):
                return True
        return is_backend_healthy(f"{protocol}://{host}:{port}")
    except Exception:
        return False


def format_markdown_response(text: str) -> str:
    """Universal text formatter that intelligently structures any text response.
    
    This is a general solution that:
    - Works with any text format (raw markdown, plain text, mixed, HTML)
    - Automatically detects and formats common patterns
    - Improves readability without changing content
    - Handles edge cases and dense text
    - Detects HTML responses and returns them as-is for Streamlit rendering
    """
    if not text:
        return text
    
    # Check if the response is HTML (starts with < and contains HTML tags)
    if text.strip().startswith('<') and ('<' in text and '>' in text):
        # This is an HTML response, return as-is for Streamlit to render
        return text.strip()
    
    # Step 1: Basic cleaning and escape sequence handling
    text = text.strip()
    text = text.replace('\\n\\n', '\n\n')
    text = text.replace('\\n', '\n')
    
    # Step 2: Preserve Markdown hard line breaks (two+ spaces at end of line)
    text = re.sub(r'[ \t]{2,}(?=\n|$)', '\n', text)

    # Step 3: Universal pattern detection and formatting
    text = detect_and_format_patterns(text)
    
    # Step 4: URL formatting
    text = format_urls(text)
    
    # Step 5: Structure improvement
    text = improve_structure(text)
    
    # Step 6: Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_markdown_response_streaming(text: str) -> str:
    """Lightweight, streaming-safe formatter for partial LLM outputs.

    - Only applies idempotent, low-risk transforms suitable for partial tokens
    - Converts escaped newlines to actual newlines
    - Makes bare URLs clickable without restructuring surrounding text
    - Collapses excessive newlines
    - Does NOT try to insert headers/lists or bold markers to avoid flicker
    - Detects HTML responses and returns them as-is for Streamlit rendering
    """
    if not text:
        return text

    # Check if the response is HTML (starts with < and contains HTML tags)
    if text.strip().startswith('<') and ('<' in text and '>' in text):
        # This is an HTML response, return as-is for Streamlit to render
        return text.strip()

    # Convert escape sequences emitted by models
    text = text.replace('\\n\\n', '\n\n')
    text = text.replace('\\n', '\n')

    # Preserve Markdown hard line breaks (two+ spaces at end of line)
    text = re.sub(r'[ \t]{2,}(?=\n|$)', '\n', text)

    # Make bare URLs clickable (idempotent)
    text = re.sub(r"(?<!\[)(https?://[^\s\)]+)(?!\])", r"[\1](\1)", text)

    # Clean up excessive newlines during streaming
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_and_format_patterns(text: str) -> str:
    """
    Highlight step and section headings in a response.

    The line and paragraph breaks the model writes now arrive intact, so this no longer
    tries to rebuild structure from punctuation - guessing where a list or paragraph
    started is what glued "1 200 000" into "1200000".
    """
    # Detect and format step patterns
    text = re.sub(r'(Bước \d+:)', r'**\1**', text)  # Vietnamese steps
    text = re.sub(r'(Step \d+:)', r'**\1**', text)  # English steps

    # Detect and format section headers (lines ending with colon)
    text = re.sub(r'^([A-Za-zÀ-ỹ\s]+:)$', r'**\1**', text, flags=re.MULTILINE)

    return text


def format_urls(text: str) -> str:
    """Format URLs to be clickable markdown links."""
    
    # Make URLs clickable (basic pattern) - only if not already formatted
    text = re.sub(r"(?<!\[)(https?://[^\s\)]+)(?!\])", r"[\1](\1)", text)
    
    # Fix double URL formatting issue
    text = re.sub(r"\[([^\]]+)\]\(\[([^\]]+)\]\([^)]+\)\)", r"[\1](\2)", text)
    
    # Fix malformed links like "[text]url" -> "[text](url)"
    text = re.sub(r'\[([^\]]+)\](https?://[^\s\)]+)', r'[\1](\2)', text)
    
    return text


def improve_structure(text: str) -> str:
    """
    Add breathing room around headings.

    List items are left alone: they arrive on their own lines already, and re-breaking
    them used to split numbers that merely looked like list markers.
    """
    # Ensure proper spacing around **Bước X:** patterns
    text = re.sub(r'(\*\*Bước \d+:\*\*)', r'\n\n\1\n', text)

    # Ensure proper spacing around other bold section headers
    text = re.sub(r'(\*\*[^:]+:\*\*)', r'\n\n\1\n', text)

    # Ensure proper spacing around ## headers
    text = re.sub(r'(## [^\n]+)', r'\n\n\1\n', text)

    return text


class ChatApp:
    """
    A Streamlit application for chatting with documents.
    Interface-only version: UI elements without backend wiring.
    """

    def __init__(self):
        # st.set_page_config(page_title="ChatBot", initial_sidebar_state="collapsed")
        # st.title("ChatBot")

        # Initialize session state variables only if they are not already set
        if "uploaded_docs" not in st.session_state:
            st.session_state.uploaded_docs = []
        if "vectordb" not in st.session_state:
            st.session_state.vectordb = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def _reset_session_dirs_and_state(self):
        """Interface-only reset: no filesystem operations."""
        st.toast("Session state reset (no file operations)")

    def reset_all(self):
        """Reset Streamlit session state (interface only, no file operations)."""
        # Clean project data: chunks, vectors, temp, logs
        with st.spinner("Cleaning data folders (chunks, vectors, temp, logs)..."):
            try:
                cleanup_data_folders()
                st.toast("Data folders cleaned")
            except Exception:
                st.toast("Failed to clean data folders")
        st.session_state.uploaded_docs = []
        st.session_state.vectordb = None
        st.session_state.chat_history = []
        # Reset file_uploader widget by changing its key
        st.session_state["uploaded_docs_uploader_key"] = str(time.time())
        st.toast("All session state reset")
        st.rerun()

    def run(self):
    # Thêm logo và title cùng dòng, logo bo góc tròn, size nhỏ
        logo_base64 = get_image_base64("assets/logo_tringhia.jpg")
        if logo_base64:

            # Ẩn menu, header, footer của Streamlit
            hide_streamlit_style = """
                <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    header {visibility: hidden;}
                    /* Ẩn icon 🔗 trong tiêu đề */
                    h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a {
                        display: none !important;
                    }
                    /* Fix iframe header to be fixed across entire page */
                    iframe[height="90"] {
                        position: fixed !important;
                        top: 0 !important;
                        left: 0 !important;
                        right: 0 !important;
                        width: 100vw !important;
                        z-index: 9999 !important;
                    }
                </style>
            """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)

            # Header cố định trên cùng với buttons
            st.components.v1.html(
                f"""
                <style>
                    .chat-header {{
                        position: fixed;
                        /* Respect device/browser safe area (macOS titlebar / iOS notch / mobile browser UI) */
                        top: env(safe-area-inset-top);
                        left: 0;
                        right: 0;
                        width: 100vw;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        /* Add safe-area padding so header content isn't clipped */
                        padding: calc(10px + env(safe-area-inset-top)) 15px 10px 15px;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        border-bottom: 1px solid rgba(255,255,255,0.08);
                        z-index: 9999;
                        /* Increase height to include safe-area inset */
                        height: calc(55px + env(safe-area-inset-top));
                        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
                        box-sizing: border-box;
                    }}
                    .header-left {{
                        display: flex;
                        align-items: center;
                        gap: 12px;
                    }}
                    .header-right {{
                        display: flex;
                        align-items: center;
                        gap: 6px;
                    }}
                    .chat-header img {{
                        width: 36px;
                        height: 36px;
                        border-radius: 50%;
                        object-fit: cover;
                    }}
                    .chat-header h2 {{
                        margin: 0;
                        color: white;
                        font-size: 19px;
                        font-weight: 700;
                        letter-spacing: 0.3px;
                    }}
                    .header-btn {{
                        background: transparent;
                        border: none;
                        border-radius: 8px;
                        color: rgba(255,255,255,0.8);
                        padding: 6px;
                        font-size: 12px;
                        cursor: pointer;
                        transition: all 0.2s ease;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        min-width: 32px;
                        height: 32px;
                        outline: none;
                    }}
                    .header-btn:hover {{
                        background: rgba(255,255,255,0.1);
                        color: white;
                        transform: translateY(-1px);
                    }}
                    .header-btn:active {{
                        transform: translateY(0);
                    }}
                    .header-btn:focus {{
                        outline: none;
                        box-shadow: none;
                    }}
                    /* Đẩy phần nội dung chatbot xuống để không bị header che */
                    .block-container {{
                        /* Fallback static padding then prefer safe-area aware calc */
                        padding-top: 60px !important;
                        padding-top: calc(56px + env(safe-area-inset-top)) !important;
                        padding-left: 10px !important;
                        padding-right: 15px !important;
                        padding-bottom: 10px !important;
                        max-width: 100% !important;
                    }}
                </style>

                <div class="chat-header">
                    <div class="header-left">
                        <img src="data:image/jpeg;base64,{logo_base64}" alt="Logo">
                        <h2>ChatBot</h2>
                    </div>
                    <div class="header-right">
                        <button class="header-btn fullscreen-btn" data-action="fullscreen" title="Toàn màn hình">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
                                <polyline points="14,9 21,3 21,9"></polyline>
                                <polyline points="21,15 14,21 14,15"></polyline>
                                <polyline points="10,9 3,3 3,9"></polyline>
                                <polyline points="3,15 10,21 10,15"></polyline>
                            </svg>
                        </button>
                        <button class="header-btn close-btn" data-action="close" title="Đóng tạm thời">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                </div>

                <script>
                    function toggleFullscreen() {{
                        //console.log('toggleFullscreen called');
                        
                        // Always try to use PostMessage API first for iframe fullscreen
                        if (window.top !== window) {{
                            //console.log('Requesting fullscreen via PostMessage to parent');
                            try {{
                                window.top.postMessage({{
                                    type: 'IFRAME_FULLSCREEN_REQUEST',
                                    action: 'toggle',
                                    iframeId: window.frameElement ? window.frameElement.id : null
                                }}, '*');
                                return; // Exit after sending message
                            }} catch (error) {{
                                //console.error('PostMessage failed:', error);
                            }}
                        }}
                        
                    }}

                    function closeTemporarily() {{
                        //console.log('closeTemporarily called');
                        // Send message to parent to close/minimize the iframe
                        if (window.top !== window) {{
                            try {{
                                window.top.postMessage({{
                                    type: 'IFRAME_CONTROL',
                                    action: 'close'
                                }}, '*');
                               // console.log('Sent close message to parent');
                            }} catch (e) {{
                               // console.log('Failed to send close message');
                            }}
                        }}
                    }}

                    // Auto-scroll to bottom during typing animation
                    function scrollToBottom() {{
                        var chatContainer = document.querySelector('.stChatMessageContainer');
                        if (chatContainer) {{
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }}
                        // Also scroll main container
                        window.scrollTo(0, document.body.scrollHeight);
                    }}

                    // Scroll when new content is added
                    const observer = new MutationObserver(function(mutations) {{
                        mutations.forEach(function(mutation) {{
                            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {{
                                setTimeout(scrollToBottom, 100);
                            }}
                        }});
                    }});

                    // Start observing when page loads
                    document.addEventListener('DOMContentLoaded', function() {{
                        const chatContainer = document.querySelector('.stChatMessageContainer');
                        if (chatContainer) {{
                            observer.observe(chatContainer, {{ childList: true, subtree: true }});
                        }}

                        // Bind button events using event delegation
                        document.addEventListener('click', function(event) {{
                            const button = event.target.closest('.header-btn');
                            if (!button) return;

                            const action = button.getAttribute('data-action');
                            if (action === 'fullscreen') {{
                                toggleFullscreen();
                            }} else if (action === 'close') {{
                                closeTemporarily();
                            }}
                        }});
                    }});
                </script>
                """,
                height=90
            )

            # Chat UI styling - aggressive approach to hide avatars and style messages
            chat_ui_style = """
                <style>
                /* ===== ẨN ICON AVATAR =====*/
                /* Ẩn avatar wrapper hoàn toàn */
                [data-testid="stChatMessageAvatar"] {
                    position: absolute !important;
                    opacity: 0 !important;
                    visibility: hidden !important;
                    width: 0 !important;
                    height: 0 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    pointer-events: none !important;
                    display: none !important;
                }

                /* Che phủ icon bằng cách override parent */
                div[data-testid="stChatMessage"] > div:first-child {
                    min-width: 0 !important;
                    width: 0 !important;
                    height: 0 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    display: none !important;
                }

                /* ===== STRUCTURE MESSAGES =====*/
                /* Message container - full width, no padding */
                [data-testid="stChatMessage"] {
                    width: 100% !important;
                    margin-bottom: 12px !important;
                    padding: 0 !important;
                    background: transparent !important;
                }

                /* Message content - the actual text box */
                [data-testid="stChatMessageContent"] {
                    width: 100% !important;
                    display: block !important;
                    font-size: 15px !important;
                    line-height: 1.6 !important;
                    word-wrap: break-word !important;
                }
                /* Bubble wrapper cho role-based styling */
                .bubble {
                    padding: 12px 16px !important;
                    border-radius: 12px !important;
                    word-wrap: break-word !important;
                    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15) !important;
                    max-width: 92% !important;
                    clear: both !important;
                }

                /* User bubble: bên PHẢI, lavender */
                .bubble.user {
                    background: #f5f3ff !important;
                    color: #1e1e1e !important;
                    border-radius: 12px 12px 0 12px !important;
                    box-shadow: 0 2px 8px rgba(85, 99, 209, 0.15) !important;
                    float: right !important;
                    margin-left: 8% !important;
                    margin-bottom: 8px !important;
                    font-size: 15px !important;
                }

                /* Assistant bubble: bên TRÁI, tím */
                .bubble.assistant {
                    background: linear-gradient(135deg, #5563d1 0%, #6b3fa0 100%) !important;
                    color: white !important;
                    border-radius: 12px 12px 12px 0 !important;
                    box-shadow: 0 2px 8px rgba(85, 99, 209, 0.3) !important;
                    float: left !important;
                    margin-right: 8% !important;
                    margin-bottom: 8px !important;
                    font-size: 15px !important;
                }

                /* ===== INPUT STYLING =====*/
                [data-testid="stChatInputContainer"] {
                    background: white !important;
                    padding: 4px 20px !important;
                    border-radius: 12px !important;
                    border: 1px solid #e0e7ff !important;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
                }

                [data-testid="stChatInputTextarea"] {
                    background-color: white !important;
                    color: #333 !important;
                    border: none !important;
                    border-radius: 8px !important;
                    padding: 6px 10px !important;
                    font-size: 15px !important;
                }

                /* ===== TEXT STYLING =====*/
                p, span {
                    word-break: break-word !important;
                    overflow-wrap: break-word !important;
                }

                a {
                    color: #667eea !important;
                    text-decoration: underline !important;
                }

                a:hover {
                    color: #764ba2 !important;
                }

                /* Đảm bảo text trong tin nhắn user cũng rõ */
                .bubble.user p,
                .bubble.user span,
                .bubble.user div,
                .bubble.user h1,
                .bubble.user h2,
                .bubble.user h3,
                .bubble.user h4,
                .bubble.user h5,
                .bubble.user h6,
                .bubble.user strong,
                .bubble.user b {
                    color: #1e1e1e !important;
                }

                .bubble.user a {
                    color: #667eea !important;
                }

                /* Đảm bảo text và link luôn hiển thị trong tin nhắn bot */
                .bubble.assistant p,
                .bubble.assistant span,
                .bubble.assistant div,
                .bubble.assistant h1,
                .bubble.assistant h2,
                .bubble.assistant h3,
                .bubble.assistant h4,
                .bubble.assistant h5,
                .bubble.assistant h6,
                .bubble.assistant strong,
                .bubble.assistant b {
                    color: white !important;
                }

                .bubble.assistant a {
                    color: #fff !important;
                    text-decoration: underline !important;
                    font-weight: 600 !important;
                }

                /* Ensure proper container width */
                .stChatMessageContainer {
                    display: flex !important;
                    flex-direction: column !important;
                    width: 100% !important;
                    padding: 15px 0 !important;
                }

                /* Typing animation cursor */
                .typing-cursor::after {
                    content: '|';
                    animation: blink 1.2s infinite;
                    color: white;
                    font-weight: bold;
                    margin-left: 2px;
                }

                @keyframes blink {
                    0%, 60% { opacity: 1; }
                    61%, 100% { opacity: 0; }
                }

                /* Typing indicator */
                .typing-indicator {
                    display: flex;
                    align-items: center;
                    gap: 1px;
                    color: white;
                    font-style: italic;
                }

                .dots {
                    display: inline-block;
                }

                .dots::after {
                    content: '';
                    animation: dots 1.5s infinite;
                    font-size: 16px;
                }

                @keyframes dots {
                    0%, 20% { content: ''; }
                    40% { content: '.'; }
                    60% { content: '..'; }
                    80%, 100% { content: '...'; }
                }

                /* Fullscreen fixes */
                :fullscreen {
                    background: #ffffff !important;
                }

                :fullscreen body {
                    background: #ffffff !important;
                    color: #000000 !important;
                }

                :fullscreen .stApp {
                    background: #ffffff !important;
                    display: block !important;
                    visibility: visible !important;
                    width: 100vw !important;
                    height: 100vh !important;
                }

                :fullscreen .main {
                    display: block !important;
                    visibility: visible !important;
                }

                :fullscreen .block-container {
                    display: block !important;
                    visibility: visible !important;
                }

                :fullscreen [data-testid="stChatMessage"] {
                    display: block !important;
                    visibility: visible !important;
                }

                :fullscreen .stChatMessageContainer {
                    display: block !important;
                    visibility: visible !important;
                }

                :fullscreen * {
                    opacity: 1 !important;
                    visibility: visible !important;
                    color: inherit !important;
                    display: block !important;
                }

                /* CSS-based fullscreen simulation */
                .fullscreen-mode {
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100vw !important;
                    height: 100vh !important;
                    z-index: 9999 !important;
                    background: white !important;
                    overflow: auto !important;
                }

                .fullscreen-mode .stApp {
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100vw !important;
                    height: 100vh !important;
                    z-index: 9999 !important;
                    background: white !important;
                    overflow: auto !important;
                }

                .fullscreen-mode iframe {
                    z-index: 10000 !important;
                }

                .fullscreen-mode * {
                    visibility: visible !important;
                    opacity: 1 !important;
                    color: inherit !important;
                }

                .fullscreen-mode .block-container {
                    /* Reserve space for the overlay header in simulated fullscreen. Use fallback then safe-area-aware calc. */
                    padding-top: 56px !important;
                    padding-top: calc(56px + env(safe-area-inset-top)) !important;
                }
                
                section .block-container {
                    padding: 2rem 1rem 1rem !important;
                }
                
                section .st-emotion-cache-qdbtli.ea3mdgi2 {
                    padding: 1rem 1rem 20px !important;
                }
                
                </style>
            """
            st.markdown(chat_ui_style, unsafe_allow_html=True)
        else:
            st.header("ChatBot")
        # Initialize API base URL and optionally auto-start backend regardless of sidebar visibility
        # Check if SSL certificates exist to determine protocol
        ssl_cert_file = "./SSL/fullchain.pem"
        ssl_key_file = "./SSL/privkey_converted.pem"
        protocol = "https" if os.path.exists(ssl_cert_file) and os.path.exists(ssl_key_file) else "http"
        DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", f"{protocol}://127.0.0.1:8504")
        api_base_url = DEFAULT_API_BASE_URL
        if os.getenv("AUTO_START_API", "true").lower() == "false" and not is_backend_healthy(api_base_url):
            try:
                # Parse URL to extract host and port
                if api_base_url.startswith("https://"):
                    host_port = api_base_url.replace("https://", "").split(":")
                else:
                    host_port = api_base_url.replace("http://", "").split(":")
                host = host_port[0]
                port = int(host_port[1]) if len(host_port) > 1 else 8504
                with st.spinner("Starting backend API..."):
                    started = start_backend_subprocess(host, port, protocol)
                    if started:
                        st.toast("Backend started")
                    else:
                        st.toast("Could not auto-start backend. Start it manually.")
            except Exception:
                st.toast("Auto-start attempt failed. Start the API manually.")

        # Sidebar hidden by default
        show_sidebar = False
        # Sidebar controlled via env only (no visible toggles)
        # show_sidebar = os.getenv("SHOW_SIDEBAR", "true").lower() == "true"
        if show_sidebar:
            with st.sidebar:
                # Startup handled above; sidebar UI only

                st.subheader("Documents")
                uploader_key = st.session_state.get("uploaded_docs_uploader_key", "uploaded_docs_uploader")
                uploaded_docs = st.file_uploader(
                    "Upload (.pdf, .txt, .docx, .csv, .xlsx)",
                    type=["pdf", "txt", "docx", "csv", "xlsx"],
                    accept_multiple_files=True,
                    key=uploader_key
                )

                if uploaded_docs:
                    pass

                if st.button(" Process Inputs", use_container_width=True):
                    if not is_backend_healthy(api_base_url):
                        st.toast("Backend not reachable. Start FastAPI and try again.")
                    else:
                        # Upload files
                        if uploaded_docs:
                            files_payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploaded_docs]
                            try:
                                resp = requests.post(f"{api_base_url}/files/upload", files=files_payload, timeout=600, verify=False)
                                if resp.ok:
                                    st.session_state.uploaded_docs.extend([f.name for f in uploaded_docs])
                                    st.toast("Files uploaded")
                                else:
                                    st.toast(f"Upload failed: {resp.status_code}")
                            except Exception as e:
                                st.toast(f"Upload error: {e}")

                if os.getenv("ENABLE_MAINTENANCE", "true").lower() == "true":
                    st.markdown("---")
                    st.subheader("Maintenance")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        if st.button("Reset", use_container_width=True):
                            self.reset_all()
                    with col_m2:
                        if st.button("Refresh Vector", use_container_width=True):
                            with st.spinner("Refreshing knowledge base..."):
                                st.session_state.vectordb = "placeholder"
                                st.toast("Knowledge base refreshed")
        else:
            # Sidebar hidden by configuration
            pass

        # -------------------------------
        # Vectorstore and Chat (interface only)
        # -------------------------------
        has_any_inputs = bool(st.session_state.uploaded_docs)

        if has_any_inputs and st.session_state.vectordb is None:
            with st.spinner("Updating knowledge base..."):
                st.session_state.vectordb = "placeholder"
                st.toast("Knowledge base is ready")
                # Gửi tin nhắn chào mừng lần đầu tiên
        if len(st.session_state.chat_history) == 0:
            greeting_message = """<strong>Xin chào, Bạn cần tư vấn gì hôm nay? </strong>"""
            st.session_state.chat_history.append({"role": "assistant", "content": greeting_message})

        # Render chat history
        for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    # Wrap content với class dựa trên role
                    bubble_class = "user" if msg["role"] == "user" else "assistant"
                    st.markdown(f'<div class="bubble {bubble_class}">{msg["content"]}</div>', unsafe_allow_html=True)

        # Chat input (streams from backend /chat)
        user_input = st.chat_input("Nhập câu hỏi của bạn")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                # Wrap user input
                st.markdown(f'<div class="bubble user">{user_input}</div>', unsafe_allow_html=True)

            with st.chat_message("assistant"):
                # Show typing indicator first
                placeholder = st.empty()
                placeholder.markdown('<div class="bubble assistant"><div class="typing-indicator">Chatbot đang xử lý yêu cầu của bạn<span class="dots"></span></div></div>', unsafe_allow_html=True)
                
                accumulated = ""
                stream_error = None
                last_flush = time.time()
                try:
                    # Use api_base_url from sidebar and pass only the most recent 10 messages
                    recent_history = [msg for msg in st.session_state.chat_history[-10:] if msg["role"] in ["user", "assistant"]] if st.session_state.chat_history else []
                    for kind, value in stream_chat(api_base_url, user_input, recent_history):
                        if kind == "error":
                            stream_error = value
                            break
                        accumulated += value
                        now = time.time()
                        # Flush frequently for smooth ChatGPT-like typing effect
                        if value.endswith((" ", "\n", ".", ",", ":", ";", "!", "?")) or (now - last_flush) > 0.05:
                            # Wrap streaming content with typing cursor
                            placeholder.markdown(f'<div class="bubble assistant typing-cursor">{format_markdown_response_streaming(accumulated)}</div>', unsafe_allow_html=True)
                            last_flush = now
                        # Small delay for natural typing rhythm
                        time.sleep(0.02)
                except Exception as e:
                    stream_error = str(e)

                if stream_error is not None:
                    # Errors are shown as a UI notice and kept out of chat_history: stored
                    # there they would render as something the bot said, and get sent back
                    # to the model as context on the next turn.
                    placeholder.empty()
                    st.error(f"Không nhận được câu trả lời: {stream_error}")
                else:
                    # After full completion, apply full formatter and update the UI immediately
                    final_text = format_markdown_response(accumulated)
                    # Wrap final content without typing cursor
                    placeholder.markdown(f'<div class="bubble assistant">{final_text}</div>', unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_text})


if __name__ == "__main__":
    ChatApp().run()
