"""Temporary Gradio showcase for the first three MuseTalk avatars.

Run from the repository root with: python gradio_demo/app.py
"""

from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "avatar_image"
VIDEO_DIR = ROOT / "data" / "public"
INPUT_VIDEO_DIR = ROOT / "data" / "input" / "video"

AVATARS = [
    {
        "name": "中文讲解员",
        "tag": "清晰 · 稳重",
        "image": IMAGE_DIR / "chinese.png",
        "video": VIDEO_DIR / "chinese2.mp4",
        "intro": "适合产品介绍、知识讲解和欢迎词。",
    },
    {
        "name": "商务男士",
        "tag": "专业 · 可信",
        "image": IMAGE_DIR / "商务男1.png",
        "video": VIDEO_DIR / "商务男确定.mp4",
        "intro": "适合企业宣传、会议主持和销售场景。",
    },
    {
        "name": "沉稳中年",
        "tag": "亲和 · 从容",
        "image": IMAGE_DIR / "中年.png",
        "video": INPUT_VIDEO_DIR / "陈屿_low.mp4",
        "intro": "适合访谈、故事表达和陪伴式内容。",
    },
]


def choose_avatar(evt: gr.SelectData):
    index = int(evt.index or 0)
    index = max(0, min(index, len(AVATARS) - 1))
    avatar = AVATARS[index]
    return (
        avatar["image"],
        f"{avatar['name']}  ·  {avatar['tag']}",
        avatar["intro"],
        index,
        f"已选择：{avatar['name']}",
    )


def start_show(index: int, audio):
    index = int(index or 0)
    avatar = AVATARS[max(0, min(index, len(AVATARS) - 1))]
    source = avatar["video"] if avatar["video"].exists() else None
    audio_note = "已载入音频，可接入推理服务生成新视频。" if audio else "当前播放预置演示片段。"
    return source, f"正在展示「{avatar['name']}」 · {audio_note}"


gallery_items = [(str(a["image"]), f"{a['name']}  |  {a['tag']}") for a in AVATARS]

CSS = """
:root { --ink:#17221f; --muted:#687570; --mint:#a7e6cb; --lime:#d9f36a; --paper:#f5f6ef; }
body { background: var(--paper); }
.gradio-container { max-width: 1180px !important; margin: 0 auto; }
.hero { padding: 34px 4px 20px; }
.eyebrow { color:#477466; font: 700 12px/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing:.18em; text-transform:uppercase; }
.hero h1 { color:var(--ink); font: 800 clamp(34px,6vw,66px)/.98 Georgia, serif; letter-spacing:-.045em; margin:10px 0 14px; }
.hero p { color:var(--muted); font-size:16px; max-width:590px; margin:0; }
.panel { background:rgba(255,255,255,.72); border:1px solid #dce5df; border-radius:24px; padding:20px; box-shadow:0 15px 40px rgba(43,73,62,.07); }
.panel h3 { color:var(--ink); margin:0 0 5px; }
.gallery .grid-wrap { gap:14px !important; }
.gallery .thumbnail-item { border-radius:16px !important; overflow:hidden; border:2px solid transparent !important; transition:.2s; }
.gallery .thumbnail-item:hover { transform:translateY(-3px); }
button.primary { background:var(--ink) !important; border:0 !important; color:#fff !important; }
.status { color:#477466; font-weight:600; }
footer { display:none !important; }
"""

with gr.Blocks(title="MuseTalk · 数字人展示", css=CSS, theme=gr.themes.Soft(primary_hue="green")) as demo:
    selected = gr.State(0)
    gr.HTML(
        '<div class="hero"><div class="eyebrow">MUSE TALK / QUICK SHOWCASE</div>'
        '<h1>让每一句话，<br>都有一个面孔。</h1>'
        '<p>一个临时的数字人展示台。选择形象，载入声音，马上预览属于你的表达。</p></div>'
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="panel"):
            gr.Markdown("### 选择数字人\n点击下方任意卡片开始")
            gallery = gr.Gallery(gallery_items, columns=3, rows=1, height=350, object_fit="cover", show_label=False, elem_classes="gallery")
            name = gr.Markdown("中文讲解员  ·  清晰 · 稳重")
            intro = gr.Markdown(AVATARS[0]["intro"])
            audio = gr.Audio(label="可选：上传一段语音", type="filepath", sources=["upload", "microphone"])
            start = gr.Button("开始展示  →", variant="primary", elem_classes="primary", size="lg")
            status = gr.Markdown("已选择：中文讲解员", elem_classes="status")
        with gr.Column(scale=7, elem_classes="panel"):
            gr.Markdown("### 实时预览")
            preview = gr.Video(value=str(AVATARS[0]["video"]), label="", autoplay=True, loop=True, show_label=False, height=520)
            gr.Markdown("预置片段用于快速预览；接入推理服务后，可使用左侧音频生成新视频。")

    gallery.select(choose_avatar, outputs=[gr.Image(visible=False), name, intro, selected, status])
    start.click(start_show, inputs=[selected, audio], outputs=[preview, status])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
