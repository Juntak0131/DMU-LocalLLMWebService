# ============================================================
# [실습 1] 로컬 LLM 기반 개인정보 문서 정리 챗봇
# 방식: 문서 텍스트를 통째로 추출해 프롬프트에 그대로 삽입 (RAG 축소판)
#
# 팀 역할 분담:
#   - [백엔드] : 문서 텍스트 추출 + Ollama 호출 + 원문 대조 검증
#   - [프론트] : Gradio UI
#   - [PM]     : 맨 아래 발표 포인트 참고
#
# 실행 전 준비:
#   1) ollama run exaone3.5:2.4b   (최초 1회, 받은 후 /bye로 종료)
#   2) pip install ollama gradio pypdf
#   3) python app.py
# ============================================================

import re
import json
import ollama
import gradio as gr
from pypdf import PdfReader

MODEL_NAME = "exaone3.5:2.4b"


# ------------------------------------------------------------
# [백엔드 1] 문서 텍스트 추출 (PDF든 TXT든 전체를 하나의 문자열로)
# ------------------------------------------------------------
def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


# ------------------------------------------------------------
# [백엔드 2] 문서 원문을 통째로 프롬프트에 삽입해 개인정보 항목 추출 요청
# → LLM에게 "새로 만들지 말고, 원문 내용만 그대로 정리하라"고 강하게 제한
# ------------------------------------------------------------
def extract_personal_info(document_text: str) -> dict:
    prompt = f"""아래는 문서 원문 전체입니다. 이 문서에서 개인정보 항목만 찾아
반드시 JSON 형식으로만 응답하세요. 다른 설명은 절대 붙이지 마세요.

규칙:
- 문서 원문에 실제로 등장하는 값만 적으세요. 절대로 값을 새로 만들거나 추측하지 마세요.
- 해당 정보가 문서에 없으면 값에 "없음"이라고 적으세요.
- 아래 JSON 형식을 정확히 지키세요.

{{
  "이름": "",
  "연락처": "",
  "이메일": "",
  "주소": "",
  "생년월일": ""
}}

[문서 원문]
{document_text}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response["message"]["content"].strip()

    # 모델이 ```json 코드블럭으로 감싸는 경우 대비
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 파싱 실패 시 원본 텍스트라도 보여주기 위한 안전장치
        return {"파싱오류": raw}


# ------------------------------------------------------------
# [백엔드 3] 정확도 검증 — 추출된 값이 원문에 실제로 존재하는지 대조
# → 이 부분이 "정확도 문제없는 개발" 요건의 핵심 근거
# ------------------------------------------------------------
def verify_against_source(info: dict, document_text: str) -> dict:
    verified = {}
    for key, value in info.items():
        if value in ("없음", "", None):
            verified[key] = "없음"
        elif value in document_text:
            verified[key] = value  # 원문에 실제로 존재 → 신뢰
        else:
            verified[key] = f"⚠ 확인 필요 (원문에서 발견되지 않음): {value}"
    return verified


# ------------------------------------------------------------
# [백엔드 4] 파이프라인 연결
# ------------------------------------------------------------
def run_pipeline(file):
    if file is None:
        return "파일을 업로드해주세요."

    # Gradio 6.x에서 type="filepath"를 사용하면 문자열 경로가 전달됩니다.
    file_path = file if isinstance(file, str) else file.name
    document_text = extract_text(file_path)
    if not document_text.strip():
        return "문서에서 텍스트를 추출하지 못했습니다. (스캔본 이미지일 수 있음)"

    raw_info = extract_personal_info(document_text)
    verified_info = verify_against_source(raw_info, document_text)

    result_lines = [f"- {k}: {v}" for k, v in verified_info.items()]
    return "\n".join(result_lines)


# ------------------------------------------------------------
# [프론트] Gradio UI
# ------------------------------------------------------------
# 개인정보를 다루는 서비스인 만큼 "보안 / 신뢰 / 깔끔함"을 중심으로
# UI를 구성했습니다. 기존 백엔드 함수는 그대로 사용합니다.
CUSTOM_CSS = """
:root {
    --bg: #f6f8fc;
    --card: #ffffff;
    --text: #172033;
    --muted: #667085;
    --primary: #315efb;
    --primary-dark: #2448c7;
    --border: #e6eaf2;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#hero {
    padding: 42px 44px 36px;
    margin: 18px 0 22px;
    border-radius: 24px;
    background: linear-gradient(135deg, #eef3ff 0%, #ffffff 65%);
    border: 1px solid #dfe7ff;
    box-shadow: 0 12px 35px rgba(31, 55, 100, 0.08);
}

.hero-badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: #e8eeff;
    color: #315efb;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 14px;
}

.hero-title {
    margin: 0;
    font-size: 34px;
    line-height: 1.25;
    letter-spacing: -1px;
}

.hero-subtitle {
    margin: 12px 0 0;
    color: var(--muted);
    font-size: 16px;
    line-height: 1.7;
}

.section-title {
    font-size: 18px;
    font-weight: 800;
    margin: 8px 0 10px;
}

.info-card {
    padding: 18px 20px;
    border-radius: 16px;
    background: var(--card);
    border: 1px solid var(--border);
    height: 100%;
    box-shadow: 0 6px 20px rgba(31, 55, 100, 0.05);
}

.info-card strong {
    display: block;
    margin-bottom: 7px;
    font-size: 15px;
}

.info-card span {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
}

#upload-box, #result-box {
    border-radius: 18px !important;
    border: 1px solid var(--border) !important;
    background: #fff !important;
}

#run-button {
    min-height: 52px;
    border-radius: 13px !important;
    font-size: 16px !important;
    font-weight: 800 !important;
}

.footer-note {
    text-align: center;
    color: #98a2b3;
    font-size: 12px;
    margin: 20px 0 4px;
}
"""

with gr.Blocks(
    title="Privacy Doc · 개인정보 문서 정리"
) as demo:

    gr.HTML("""
    <section id="hero">
        <div class="hero-badge">🔒 LOCAL AI · PRIVACY FIRST</div>
        <h1 class="hero-title">개인정보 문서 정리 챗봇</h1>
        <p class="hero-subtitle">
            PDF 또는 TXT 문서를 업로드하면 로컬 LLM이
            이름 · 연락처 · 이메일 · 주소 · 생년월일을 정리합니다.<br>
            문서 원문과 다시 대조하여 추출 결과를 한 번 더 검증합니다.
        </p>
    </section>
    """)

    gr.HTML('<div class="section-title">서비스 특징</div>')

    with gr.Row():
        gr.HTML("""
        <div class="info-card">
            <strong>🖥️ 로컬 처리</strong>
            <span>Ollama 기반 로컬 LLM을 사용해 개인정보 문서를 외부 API로 보내지 않습니다.</span>
        </div>
        """)
        gr.HTML("""
        <div class="info-card">
            <strong>🛡️ 원문 대조 검증</strong>
            <span>LLM이 추출한 값이 실제 문서 원문에 존재하는지 코드로 다시 확인합니다.</span>
        </div>
        """)
        gr.HTML("""
        <div class="info-card">
            <strong>📄 PDF / TXT 지원</strong>
            <span>문서 전체의 텍스트를 추출해 개인정보 항목을 한눈에 정리합니다.</span>
        </div>
        """)

    gr.Markdown("### 📄 문서 분석", elem_classes="section-title")

    with gr.Row():
        with gr.Column(scale=5):
            file_input = gr.File(
                label="분석할 문서",
                file_types=[".pdf", ".txt"],
                type="filepath",
                elem_id="upload-box"
            )
            gr.Markdown(
                "PDF 또는 TXT 파일을 올려주세요. 실제 개인정보 대신 **가짜 데이터가 포함된 샘플 문서** 사용을 권장합니다."
            )

        with gr.Column(scale=2):
            run_btn = gr.Button(
                "🔍 개인정보 정리하기",
                variant="primary",
                elem_id="run-button"
            )

    gr.Markdown("### 📋 분석 결과", elem_classes="section-title")

    output_box = gr.Textbox(
        label="추출 및 검증 결과",
        placeholder="문서를 업로드하고 '개인정보 정리하기' 버튼을 눌러주세요.",
        lines=10,
        elem_id="result-box"
    )

    gr.HTML("""
    <div class="footer-note">
        🔐 Privacy Doc · Ollama + Gradio · 모든 처리는 로컬 환경에서 수행됩니다.
    </div>
    """)

    run_btn.click(
        fn=run_pipeline,
        inputs=[file_input],
        outputs=[output_box]
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Pretendard")
        ),
        css=CUSTOM_CSS
    )


# ============================================================
# [PM] 발표용 포인트 정리
#
# Q. 왜 로컬 환경 개발이 필요한가?
# A. 이력서·신청서 등 이 서비스가 다루는 문서에는 이름, 연락처, 주소 같은
#    민감한 개인정보가 원문 그대로 포함되어 있다. 이를 외부 클라우드 API로
#    전송하는 것은 개인정보보호 관점에서 위험하므로, 인터넷 연결 없이
#    로컬 PC 내부에서만 처리되는 Ollama 기반 LLM을 사용했다.
#
# Q. 정확도 문제는 어떻게 없앴는가?
# A. 문서 원문 전체를 프롬프트에 그대로 삽입해 LLM이 임의로 지어내지
#    않고 원문 안에서만 답하도록 제한했다. 여기서 한 단계 더 나아가,
#    LLM이 추출한 값이 실제로 원문에 존재하는 문자열인지 코드로
#    재검증(verify_against_source)하여, 원문에 없는 값은 "확인 필요"로
#    표시함으로써 잘못된 정보가 그대로 결과에 노출되지 않도록 했다.
# ============================================================
