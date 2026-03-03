#!/usr/bin/env python3
"""
transcribe_web.py (fixed)

Local web interface (runs on your computer) to:
- Upload a video/audio file
- Transcribe to transcript.txt + transcript.json (Format C)
- Optionally summarize to summary.md + summary.json (Format A)

Fixes vs original:
- Episode number is AUTO-INFERRED from the uploaded filename (e.g., "01_myvideo.mp4" -> episode 1)
- Optional manual episode override in the UI
This keeps your HTML player mapping consistent: 01_summary.json / 01_transcript.json always match Episode 1.
"""

import re
import tempfile
from pathlib import Path
import json
import gradio as gr

from transcribe_upgrade import process_one


def infer_episode_from_filename(name: str) -> int | None:
    """Infer episode number from a filename.
    Prefers leading digits like '01_' or '1-' or '002 '.
    Falls back to first 1-3 digit sequence anywhere.
    """
    if not name:
        return None
    m = re.match(r"^\s*(\d{1,3})\D", name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    m = re.search(r"(\d{1,3})", name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def run_job(file_obj, whisper_model, device, do_summary, ollama_model, ollama_url, chunk_seconds, episode_override):
    if file_obj is None:
        return "No file uploaded.", None, None, None

    # file_obj is a temp path from Gradio
    in_path = Path(file_obj)
    out_dir = Path(tempfile.mkdtemp(prefix="transcribe_out_"))

    inferred = infer_episode_from_filename(in_path.name) or 1
    ep = int(episode_override) if episode_override and int(episode_override) > 0 else inferred

    # Use the stem as a human title; episode number controls output filenames/metadata
    res = process_one(
        input_path=in_path,
        out_dir=out_dir,
        whisper_model=whisper_model,
        device=device,
        do_summary=do_summary,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        chunk_seconds=int(chunk_seconds),
        force=True,
        episode_num=ep,
        title=in_path.stem,
    )

    transcript_json = res.get("transcript_json")
    transcript_txt = res.get("transcript_txt")
    summary_json = res.get("summary_json")
    summary_md = res.get("summary_md")

    msg = "✅ Done.\n\n" + json.dumps(res, indent=2)
    msg += f"\n\nEpisode used: {ep} (inferred: {inferred})"

    return msg, transcript_json, transcript_txt, (summary_json or summary_md)


with gr.Blocks(title="Video → Transcript JSON/TXT") as demo:
    gr.Markdown("# Video → Transcript (JSON/TXT) + Optional Summary\nRuns locally on your computer.")

    with gr.Row():
        file_in = gr.File(label="Upload video/audio", file_count="single", type="filepath")

    with gr.Row():
        whisper_model = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3"], value="small", label="Whisper model")
        device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")

    with gr.Row():
        episode_override = gr.Number(value=0, precision=0, label="Episode override (0 = auto from filename)")

    with gr.Row():
        do_summary = gr.Checkbox(value=False, label="Also generate summary (requires Ollama running)")
        ollama_model = gr.Textbox(value="llama3.1:8b", label="Ollama model")
        ollama_url = gr.Textbox(value="http://localhost:11434/api/generate", label="Ollama URL")
        chunk_seconds = gr.Slider(120, 1800, value=600, step=60, label="Chunk seconds")

    run_btn = gr.Button("Transcribe")

    status = gr.Textbox(label="Status", lines=10)
    out_json = gr.File(label="Download transcript.json (Format C)")
    out_txt = gr.File(label="Download transcript.txt")
    out_summary = gr.File(label="Download summary.json or summary.md (if enabled)")

    run_btn.click(
        fn=run_job,
        inputs=[file_in, whisper_model, device, do_summary, ollama_model, ollama_url, chunk_seconds, episode_override],
        outputs=[status, out_json, out_txt, out_summary],
    )

if __name__ == "__main__":
    demo.launch()
