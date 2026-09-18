"""
Script đánh giá khách quan hiệu năng và chất lượng của model Qwen3-8B Local trên hệ thống RAG thực tế.
Chạy trực tiếp qua ChatService (Vector Store + BM25 Hybrid Search + Prompt Management + Streaming).
"""

import sys
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Thiết lập UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục gốc vào sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

from config.settings import Config
from services.chat_service import ChatService


TEST_QUERIES = [
    {
        "id": "Q1",
        "category": "Fact Retrieval (Truy xuất sự kiện)",
        "query": "Dịch vụ gia công phần mềm của TNT có những thế mạnh gì và sử dụng các ngôn ngữ lập trình nào?",
        "expected_keywords": ["15 năm", "Mỹ", "Nhật", "Singapore", ".NET", "PHP", "Java", "Python"]
    },
    {
        "id": "Q2",
        "category": "Procedure / Process (Quy trình nghiệp vụ)",
        "query": "Quy trình gia công phần mềm tại Trí Nghĩa (TNT) gồm những bước nào và tuân theo tiêu chuẩn gì?",
        "expected_keywords": ["PMI", "PMP", "Xác định phạm vi", "Xác định yêu cầu", "Xây dựng hệ thống", "Triển khai", "Nghiệm thu", "Vận hành"]
    },
    {
        "id": "Q3",
        "category": "Domain Solution (Giải pháp phần mềm)",
        "query": "Phần mềm TNM là gì và mang lại những lợi ích gì cho doanh nghiệp?",
        "expected_keywords": ["TNM", "lợi ích", "quản lý", "doanh nghiệp"]
    }
]


async def run_evaluation():
    print("=" * 80)
    print(" BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG & CHẤT LƯỢNG RAG VỚI MODEL LOCAL QWEN3-8B")
    print("=" * 80)
    print(f"• LLM Base URL : {Config.LLM.OPENAI_BASE_URL() or 'OpenAI Cloud'}")
    print(f"• Model Name   : {Config.LLM.OPENAI_MODEL()}")
    print(f"• Max Context  : {Config.LLM.MAX_CONTEXT_LENGTH()} chars")
    print(f"• Temperature  : {Config.LLM.OPENAI_TEMPERATURE()}")
    print("=" * 80)

    # 1. Khởi tạo ChatService
    print("\n[1/3] Khởi tạo ChatService & Kiểm tra Knowledge Base...")
    t0 = time.time()
    chat_service = ChatService()
    init_time = time.time() - t0

    kb_status = chat_service.get_knowledge_base_status()
    doc_count = kb_status.get("document_count", 0)
    emb_count = kb_status.get("embedding_count", 0)
    print(f" [OK] Khởi tạo hoàn tất trong {init_time:.2f}s")
    print(f" • Số documents/chunks trong Vector Store : {doc_count}")
    print(f" • Số embeddings đã index                  : {emb_count}")
    print(f" • Trạng thái Knowledge Base              : {kb_status.get('status')}")

    if doc_count == 0:
        print("⚠️ Cảnh báo: Vector Store hiện đang trống. RAG sẽ không có context để đối chiếu.")

    # 2. Thực hiện đánh giá từng câu hỏi
    print("\n[2/3] Bắt đầu kiểm thử các câu hỏi thực tế từ Knowledge Base...")
    results = []

    for item in TEST_QUERIES:
        q_id = item["id"]
        cat = item["category"]
        query = item["query"]
        expected = item["expected_keywords"]

        print("\n" + "-" * 80)
        print(f" Câu hỏi [{q_id}] ({cat}):")
        print(f"   \"{query}\"")
        print("-" * 80)
        print(" Phản hồi từ Qwen3-8B (Streaming):")

        start_time = time.time()
        first_token_time = None
        chunks = []

        try:
            async for token in chat_service.stream_chat_with_memory(query):
                if first_token_time is None:
                    first_token_time = time.time()
                chunks.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()

            end_time = time.time()
            total_duration = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else total_duration
            full_text = "".join(chunks)
            token_count = len(chunks)
            tokens_per_sec = (token_count / (total_duration - ttft)) if (total_duration - ttft) > 0 else 0

            # Kiểm tra độ khớp từ khóa từ context (Grounding check)
            matched_keywords = [kw for kw in expected if kw.lower() in full_text.lower()]
            grounding_score = (len(matched_keywords) / len(expected)) * 100 if expected else 100

            print("\n" + "." * 80)
            print(f" THÔNG SỐ:")
            print(f"   • TTFT (Time to First Token) : {ttft:.2f}s (Bao gồm Hybrid Search + LLM Prompt processing)")
            print(f"   • Thời gian sinh token        : {total_duration - ttft:.2f}s")
            print(f"   • Tổng thời gian phản hồi    : {total_duration:.2f}s")
            print(f"   • Số token / chunks sinh ra   : {token_count}")
            print(f"   • Tốc độ sinh token thực tế  : {tokens_per_sec:.1f} tokens/giây")
            print(f"   • Độ khớp thông tin kỳ vọng : {len(matched_keywords)}/{len(expected)} từ khóa ({grounding_score:.0f}%) -> {matched_keywords}")

            results.append({
                "id": q_id,
                "category": cat,
                "ttft": ttft,
                "duration": total_duration,
                "tokens": token_count,
                "speed": tokens_per_sec,
                "grounding_score": grounding_score,
                "success": True,
                "response_length": len(full_text)
            })

        except Exception as e:
            print(f"\n❌ Lỗi khi xử lý câu hỏi [{q_id}]: {e}")
            results.append({
                "id": q_id,
                "category": cat,
                "success": False,
                "error": str(e)
            })

    # 3. Báo cáo tổng kết
    print("\n" + "=" * 80)
    print(" BẢNG TỔNG HỢP ĐÁNH GIÁ KHÁCH QUAN")
    print("=" * 80)
    successful = [r for r in results if r.get("success")]
    if successful:
        avg_ttft = sum(r["ttft"] for r in successful) / len(successful)
        avg_speed = sum(r["speed"] for r in successful) / len(successful)
        avg_grounding = sum(r["grounding_score"] for r in successful) / len(successful)
        avg_duration = sum(r["duration"] for r in successful) / len(successful)

        print(f"• Tỷ lệ thành công           : {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.0f}%)")
        print(f"• TTFT trung bình            : {avg_ttft:.2f}s")
        print(f"• Tốc độ sinh token TB       : {avg_speed:.1f} tokens/giây")
        print(f"• Thời gian phản hồi TB      : {avg_duration:.2f}s / câu hỏi")
        print(f"• Độ chính xác dữ liệu (RAG) : {avg_grounding:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
