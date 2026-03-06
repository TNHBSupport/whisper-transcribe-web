#!/usr/bin/env python3
"""
transcribe_web_progress.py

Local web interface to:
- Upload a video/audio file
- Transcribe to transcript.txt + transcript.json (Format C)
- Optionally summarize to summary.md + summary.json (Format A)
- Show live progress bar + streaming logs in UI

This file is self-contained and does not depend on transcribe_upgrade.py.
"""

import html as html_lib
import gc
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import fcntl
import resource
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import ffmpeg
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: ffmpeg-python. Install with:\n"
        "  python3 -m pip install -r requirements_transcribe_web_progress.txt"
    ) from e

try:
    import gradio as gr
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: gradio. Install with:\n"
        "  python3 -m pip install -r requirements_transcribe_web_progress.txt"
    ) from e

try:
    import requests
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: requests. Install with:\n"
        "  python3 -m pip install -r requirements_transcribe_web_progress.txt"
    ) from e

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: faster-whisper. Install with:\n"
        "  python3 -m pip install -r requirements_transcribe_web_progress.txt"
    ) from e

FFMPEG_THREADS = "2"
MAX_CPU_THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))
_ACTIVE_JOB_LOCK = threading.Lock()
# Optional safety mode. Set WSL_SAFE_MODE=1 to force CPU.
WSL_SAFE_MODE = os.environ.get("WSL_SAFE_MODE", "0") == "1"
WHISPER_CUDA_COMPUTE_TYPE = os.environ.get("WHISPER_CUDA_COMPUTE_TYPE", "int8_float16")
LOG_ROOT_DIR = Path(os.environ.get("TRANSCRIBE_LOG_DIR", "/mnt/e/transcribe_logs"))
TRANSCRIBE_CHUNK_SECONDS = max(300, int(os.environ.get("TRANSCRIBE_CHUNK_SECONDS", "1800")))
INSTANCE_LOCK_PATH = Path(os.environ.get("TRANSCRIBE_INSTANCE_LOCK", "/tmp/transcribe_web_progress.lock"))
_INSTANCE_LOCK_FP = None
TRANSCRIBE_MAX_RAM_GB = float(os.environ.get("TRANSCRIBE_MAX_RAM_GB", "0"))
TRANSCRIBE_NICE = int(os.environ.get("TRANSCRIBE_NICE", "0"))


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def acquire_instance_lock() -> None:
    global _INSTANCE_LOCK_FP
    INSTANCE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fp = INSTANCE_LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        fp.close()
        raise SystemExit(
            "Another transcribe_web_progress.py instance is already running.\n"
            "Stop the other process first, or set TRANSCRIBE_INSTANCE_LOCK to a different path."
        ) from e
    fp.seek(0)
    fp.truncate(0)
    fp.write(f"pid={os.getpid()}\nstarted={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    fp.flush()
    _INSTANCE_LOCK_FP = fp


def apply_runtime_limits() -> None:
    # Optional guardrail: cap this process memory so WSL is less likely to OOM-kill the whole VM.
    if TRANSCRIBE_MAX_RAM_GB > 0:
        limit_bytes = int(TRANSCRIBE_MAX_RAM_GB * 1024 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        print(f"[startup] RLIMIT_AS set to {TRANSCRIBE_MAX_RAM_GB:.2f} GB")
    if TRANSCRIBE_NICE != 0:
        os.nice(TRANSCRIBE_NICE)
        print(f"[startup] nice={TRANSCRIBE_NICE}")


def get_media_duration_seconds(path: str) -> float:
    try:
        probe = ffmpeg.probe(path)
        return float(probe["format"]["duration"])
    except Exception:
        return 0.0


def extract_audio(input_path: str, wav_path: str) -> None:
    (
        ffmpeg.input(input_path)
        .output(wav_path, ac=1, ar=16000, format="wav")
        .global_args("-threads", FFMPEG_THREADS, "-nostdin")
        .overwrite_output()
        .run(quiet=True)
    )


def extract_audio_range(input_path: str, wav_path: str, start_sec: float, duration_sec: float) -> None:
    (
        ffmpeg.input(input_path, ss=max(0.0, float(start_sec)), t=max(0.1, float(duration_sec)))
        .output(wav_path, ac=1, ar=16000, format="wav")
        .global_args("-threads", FFMPEG_THREADS, "-nostdin")
        .overwrite_output()
        .run(quiet=True)
    )


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def transcribe_segments(
    wav_path: str,
    model_size: str = "small",
    device: str = "cpu",
    model: Optional[WhisperModel] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict], str]:
    own_model = False
    if model is None:
        device = resolve_device(device, log_cb=log_cb)
        compute_type = "int8"
        if device == "cuda":
            # Set WHISPER_CUDA_COMPUTE_TYPE=float16 for max speed/quality on strong GPUs.
            compute_type = WHISPER_CUDA_COMPUTE_TYPE
        if log_cb:
            log_cb(f"[whisper] loading model={model_size} device={device} compute_type={compute_type}")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=MAX_CPU_THREADS,
            num_workers=1,
        )
        own_model = True

    total_sec = get_media_duration_seconds(wav_path)
    if log_cb:
        log_cb(f"[whisper] wav duration={total_sec:.1f}s")

    segments_iter, info = model.transcribe(
        wav_path,
        vad_filter=True,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
    )

    segs: List[Dict] = []
    last_end = 0.0
    last_emit = time.time()

    for s in segments_iter:
        start = float(getattr(s, "start", 0.0) or 0.0)
        end = float(getattr(s, "end", 0.0) or 0.0)
        text = (getattr(s, "text", "") or "").strip()

        if text:
            segs.append({"start": start, "end": end, "text": text})

        if total_sec > 0:
            last_end = max(last_end, end)
            now = time.time()
            if now - last_emit >= 0.2:
                pct = min(100.0, (last_end / total_sec) * 100.0)
                if progress_cb:
                    progress_cb(pct)
                if log_cb:
                    log_cb(f"[whisper] {hhmmss(last_end)} / {hhmmss(total_sec)} ({pct:.1f}%)")
                last_emit = now

    if total_sec > 0 and progress_cb:
        progress_cb(100.0)

    if log_cb:
        log_cb(f"[whisper] completed segments={len(segs)} language={getattr(info, 'language', 'unknown')}")

    lang = getattr(info, "language", "unknown")
    if own_model:
        # Ensure ctranslate2 buffers are released quickly on long runs.
        del model
        gc.collect()
    return segs, lang


def resolve_device(requested_device: str, log_cb: Optional[Callable[[str], None]] = None) -> str:
    if WSL_SAFE_MODE and requested_device == "cuda":
        if log_cb:
            log_cb("[whisper] WSL_SAFE_MODE=1 forcing cpu (set WSL_SAFE_MODE=0 to allow cuda)")
        return "cpu"
    if requested_device != "cuda":
        return "cpu"
    if shutil.which("nvidia-smi") is None:
        if log_cb:
            log_cb("[whisper] CUDA requested but nvidia-smi not found; falling back to cpu")
        return "cpu"
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=2,
        )
        return "cuda"
    except Exception:
        if log_cb:
            log_cb("[whisper] CUDA requested but GPU check failed; falling back to cpu")
        return "cpu"


def segments_to_txt(segs: List[Dict]) -> str:
    return "\n".join(f"[{hhmmss(s['start'])}] {s['text']}" for s in segs)


def segments_to_format_c_json(segs: List[Dict]) -> List[Dict]:
    return [{"timestamp": [round(s["start"], 3), round(s["end"], 3)], "text": s["text"]} for s in segs]


def chunk_segments_by_seconds(segs: List[Dict], chunk_seconds: int = 600) -> List[List[Dict]]:
    if not segs:
        return []

    chunks: List[List[Dict]] = []
    cur: List[Dict] = []
    window_start = segs[0]["start"]
    window_end = window_start + chunk_seconds

    for s in segs:
        if s["start"] <= window_end or not cur:
            cur.append(s)
        else:
            chunks.append(cur)
            cur = [s]
            window_start = s["start"]
            window_end = window_start + chunk_seconds

    if cur:
        chunks.append(cur)

    return chunks


def ollama_generate(prompt: str, model: str, ollama_url: str, timeout_s: int = 900) -> str:
    r = requests.post(
        ollama_url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def ollama_summarize_transcript_text(
    transcript_text: str,
    model: str,
    ollama_url: str,
    instructions: str = "",
) -> str:
    prompt = f"""{instructions}

You are an expert note-taker. Summarize the transcript below.
Return:
1) Executive summary (5-8 bullets)
2) Key topics (section headers)
3) Action items (if any)
4) Important quotes (optional)
5) 10-line ultra-short recap

TRANSCRIPT:
{transcript_text}
"""
    return ollama_generate(prompt=prompt, model=model, ollama_url=ollama_url)


def text_to_simple_html(text: str) -> str:
    esc = html_lib.escape(text)
    return f'<p style="white-space:pre-wrap; font-size:0.88rem; line-height:1.75;">{esc}</p>'


def infer_episode_from_filename(name: str) -> Optional[int]:
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


def process_one(
    input_path: Path,
    out_dir: Path,
    whisper_model: str,
    device: str,
    do_summary: bool,
    ollama_model: str,
    ollama_url: str,
    chunk_seconds: int,
    force: bool,
    episode_num: Optional[int] = None,
    title: Optional[str] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    safe_mkdir(out_dir)

    ep = episode_num if episode_num is not None else (infer_episode_from_filename(input_path.name) or 1)
    prefix = f"{ep:02d}"

    wav_path = out_dir / f"{prefix}_audio_16k_mono.wav"
    transcript_txt_path = out_dir / f"{prefix}_transcript.txt"
    transcript_json_path = out_dir / f"{prefix}_transcript.json"
    summary_md_path = out_dir / f"{prefix}_summary.md"
    summary_json_path = out_dir / f"{prefix}_summary.json"

    if not force and transcript_json_path.exists() and transcript_txt_path.exists() and (not do_summary or summary_json_path.exists()):
        if log_cb:
            log_cb("[skip] outputs already exist, skipping")
        return {
            "episode": ep,
            "title": title or input_path.stem,
            "skipped": True,
            "transcript_json": str(transcript_json_path),
            "transcript_txt": str(transcript_txt_path),
            "summary_json": str(summary_json_path) if do_summary else None,
        }

    if log_cb:
        log_cb(f"[ffmpeg] extracting audio: {input_path.name} -> {wav_path.name}")
    if progress_cb:
        progress_cb(2.0)

    input_total_sec = get_media_duration_seconds(str(input_path))
    segs: List[Dict] = []
    lang = "unknown"
    resolved_device = resolve_device(device, log_cb=log_cb)
    compute_type = "int8" if resolved_device == "cpu" else WHISPER_CUDA_COMPUTE_TYPE
    if log_cb:
        log_cb(f"[whisper] loading model={whisper_model} device={resolved_device} compute_type={compute_type}")
    model = WhisperModel(
        whisper_model,
        device=resolved_device,
        compute_type=compute_type,
        cpu_threads=MAX_CPU_THREADS,
        num_workers=1,
    )

    try:
        if input_total_sec > TRANSCRIBE_CHUNK_SECONDS:
            chunk_count = int((input_total_sec + TRANSCRIBE_CHUNK_SECONDS - 1) // TRANSCRIBE_CHUNK_SECONDS)
            if log_cb:
                log_cb(
                    f"[transcribe] long media detected ({hhmmss(input_total_sec)}), "
                    f"chunking into {chunk_count} x {TRANSCRIBE_CHUNK_SECONDS}s"
                )
            for i in range(chunk_count):
                chunk_start = float(i * TRANSCRIBE_CHUNK_SECONDS)
                chunk_dur = min(float(TRANSCRIBE_CHUNK_SECONDS), max(0.1, input_total_sec - chunk_start))
                chunk_wav = out_dir / f"{prefix}_chunk_{i+1:03d}.wav"
                if log_cb:
                    log_cb(f"[ffmpeg] chunk {i + 1}/{chunk_count} extract {hhmmss(chunk_start)} + {int(chunk_dur)}s")
                extract_audio_range(str(input_path), str(chunk_wav), start_sec=chunk_start, duration_sec=chunk_dur)

                def transcribe_progress_chunk(p: float) -> None:
                    # Allocate 8%..78% for full transcription across chunks.
                    if progress_cb:
                        overall = (i + (p / 100.0)) / max(1, chunk_count)
                        progress_cb(8.0 + 70.0 * overall)

                chunk_segs, chunk_lang = transcribe_segments(
                    str(chunk_wav),
                    model_size=whisper_model,
                    device=resolved_device,
                    model=model,
                    progress_cb=transcribe_progress_chunk,
                    log_cb=log_cb,
                )
                if lang == "unknown" and chunk_lang:
                    lang = chunk_lang
                for s in chunk_segs:
                    segs.append(
                        {
                            "start": float(s["start"]) + chunk_start,
                            "end": float(s["end"]) + chunk_start,
                            "text": s["text"],
                        }
                    )
                try:
                    chunk_wav.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            extract_audio(str(input_path), str(wav_path))

            if progress_cb:
                progress_cb(8.0)
            if log_cb:
                log_cb("[ffmpeg] extraction complete")

            def transcribe_progress(p: float) -> None:
                # Allocate 8%..78% for transcription
                if progress_cb:
                    progress_cb(8.0 + 0.70 * p)

            segs, lang = transcribe_segments(
                str(wav_path),
                model_size=whisper_model,
                device=resolved_device,
                model=model,
                progress_cb=transcribe_progress,
                log_cb=log_cb,
            )
    finally:
        del model
        gc.collect()

    transcript_txt = segments_to_txt(segs)
    transcript_txt_path.write_text(transcript_txt, encoding="utf-8")

    transcript_json = segments_to_format_c_json(segs)
    transcript_json_path.write_text(json.dumps(transcript_json, ensure_ascii=False, indent=2), encoding="utf-8")

    if log_cb:
        log_cb(f"[write] {transcript_txt_path.name}")
        log_cb(f"[write] {transcript_json_path.name}")

    result = {
        "episode": ep,
        "title": title or input_path.stem,
        "language": lang,
        "transcript_json": str(transcript_json_path),
        "transcript_txt": str(transcript_txt_path),
    }

    if not do_summary:
        if progress_cb:
            progress_cb(100.0)
        return result

    chunk_groups = chunk_segments_by_seconds(segs, chunk_seconds=chunk_seconds)
    partials = []

    if log_cb:
        log_cb(f"[summary] chunks={len(chunk_groups)} model={ollama_model}")

    total_chunks = max(1, len(chunk_groups))
    for i, group in enumerate(chunk_groups):
        chunk_text = " ".join(s["text"] for s in group).strip()
        if log_cb:
            log_cb(f"[summary] chunk {i + 1}/{total_chunks} generating")

        part = ollama_summarize_transcript_text(
            chunk_text,
            model=ollama_model,
            ollama_url=ollama_url,
            instructions=f"Chunk {i + 1}/{total_chunks}. Focus only on this section.",
        )
        partials.append(f"## Chunk {i + 1}\n{part}\n")

        if progress_cb:
            # Allocate 78%..96% for chunk summaries
            progress_cb(78.0 + 18.0 * ((i + 1) / total_chunks))

    combined = "\n\n".join(partials)

    if log_cb:
        log_cb("[summary] building final combined summary")

    final = ollama_summarize_transcript_text(
        combined,
        model=ollama_model,
        ollama_url=ollama_url,
        instructions="Now combine these chunk summaries into ONE clean final summary with no repetition.",
    )

    if progress_cb:
        progress_cb(98.0)

    summary_md_path.write_text(
        f"# Final Summary\n\n{final}\n\n---\n\n# Chunk Summaries\n\n{combined}",
        encoding="utf-8",
    )

    summary_obj = {
        "episode": ep,
        "title": title or input_path.stem,
        "summary": text_to_simple_html(final),
    }
    summary_json_path.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    if log_cb:
        log_cb(f"[write] {summary_md_path.name}")
        log_cb(f"[write] {summary_json_path.name}")

    result["summary_md"] = str(summary_md_path)
    result["summary_json"] = str(summary_json_path)

    if progress_cb:
        progress_cb(100.0)

    return result


class LiveReporter:
    def __init__(self, log_file_path: Path, ui_log_keep_lines: int = 2000) -> None:
        self._lock = threading.Lock()
        self.progress = 0.0
        self.logs: List[str] = []
        self.log_file_path = log_file_path
        self.ui_log_keep_lines = ui_log_keep_lines
        safe_mkdir(self.log_file_path.parent)
        # Line-buffered with explicit fsync in log() so diagnostics survive hard exits.
        self._fp = self.log_file_path.open("a", encoding="utf-8", buffering=1)
        self.done = False
        self.error: Optional[str] = None
        self.result: Optional[Dict] = None

    def set_progress(self, value: float) -> None:
        with self._lock:
            self.progress = max(0.0, min(100.0, float(value)))

    def log(self, line: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{stamp}] {line}"
        with self._lock:
            self.logs.append(full)
            if self.ui_log_keep_lines > 0 and len(self.logs) > self.ui_log_keep_lines:
                self.logs = self.logs[-self.ui_log_keep_lines :]
            self._fp.write(full + "\n")
            self._fp.flush()
            os.fsync(self._fp.fileno())

    def finish(self, result: Dict) -> None:
        with self._lock:
            self.result = result
            self.done = True

    def fail(self, err: str) -> None:
        with self._lock:
            self.error = err
            self.done = True

    def snapshot(self) -> Tuple[float, str, bool, Optional[str], Optional[Dict]]:
        with self._lock:
            return self.progress, "\n".join(self.logs), self.done, self.error, self.result

    def close(self) -> None:
        with self._lock:
            try:
                self._fp.close()
            except Exception:
                pass


def run_job_stream(file_obj, whisper_model, device, do_summary, ollama_model, ollama_url, chunk_seconds, episode_override):
    if not _ACTIVE_JOB_LOCK.acquire(blocking=False):
        yield 0.0, "Another transcription job is already running. Please wait for it to finish.", None, None, None, None
        return

    reporter: Optional[LiveReporter] = None
    try:
        if file_obj is None:
            yield 0.0, "No file uploaded.", None, None, None, None
            return

        in_path = Path(file_obj)
        out_dir = Path(tempfile.mkdtemp(prefix="transcribe_out_"))
        run_id = time.strftime("%Y%m%d_%H%M%S")
        log_file_path = LOG_ROOT_DIR / f"transcribe_{run_id}_{os.getpid()}.log"

        inferred = infer_episode_from_filename(in_path.name) or 1
        ep = int(episode_override) if episode_override and int(episode_override) > 0 else inferred

        reporter = LiveReporter(log_file_path=log_file_path)
        reporter.log(f"[start] input={in_path.name}")
        reporter.log(f"[start] output_dir={out_dir}")
        reporter.log(f"[start] log_file={log_file_path}")
        reporter.log(f"[start] episode={ep} (inferred={inferred})")
        reporter.log(f"[start] cpu_threads={MAX_CPU_THREADS} ffmpeg_threads={FFMPEG_THREADS}")
        reporter.log(f"[start] wsl_safe_mode={WSL_SAFE_MODE}")

        def worker() -> None:
            try:
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
                    progress_cb=reporter.set_progress,
                    log_cb=reporter.log,
                )
                reporter.log("[done] processing complete")
                reporter.finish(res)
            except Exception as e:
                reporter.log(f"[error] {e}")
                reporter.log("[traceback]\n" + traceback.format_exc())
                reporter.fail(str(e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            progress, logs, done, err, res = reporter.snapshot()

            transcript_json = res.get("transcript_json") if res else None
            transcript_txt = res.get("transcript_txt") if res else None
            summary_out = (res.get("summary_json") or res.get("summary_md")) if res else None

            if done and err:
                fail_logs = logs + "\n\nFAILED: " + err
                yield progress, fail_logs, None, None, None, str(reporter.log_file_path) if reporter else None
                return

            if done and res:
                res["session_log"] = str(reporter.log_file_path)
                final_logs = logs + "\n\n" + json.dumps(res, indent=2)
                yield 100.0, final_logs, transcript_json, transcript_txt, summary_out, str(reporter.log_file_path)
                return

            yield progress, logs, None, None, None, str(reporter.log_file_path)
            time.sleep(0.2)
    finally:
        if reporter is not None:
            reporter.log("[end] run_job_stream exiting")
            reporter.close()
        _ACTIVE_JOB_LOCK.release()


with gr.Blocks(title="Video to Transcript JSON/TXT (Live Progress)") as demo:
    gr.Markdown("# Video to Transcript (JSON/TXT) + Optional Summary\nRuns locally on your computer with live progress/log streaming.")

    with gr.Row():
        file_in = gr.File(label="Upload video/audio", file_count="single", type="filepath")

    with gr.Row():
        whisper_model = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3"], value="base", label="Whisper model")
        device = gr.Dropdown(["cpu", "cuda"], value="cpu", label="Device")

    with gr.Row():
        episode_override = gr.Number(value=0, precision=0, label="Episode override (0 = auto from filename)")

    with gr.Row():
        do_summary = gr.Checkbox(value=False, label="Also generate summary (requires Ollama running)")
        ollama_model = gr.Textbox(value="llama3.1:8b", label="Ollama model")
        ollama_url = gr.Textbox(value="http://localhost:11434/api/generate", label="Ollama URL")
        chunk_seconds = gr.Slider(120, 1800, value=600, step=60, label="Chunk seconds")

    run_btn = gr.Button("Transcribe")

    progress = gr.Slider(minimum=0, maximum=100, value=0, step=0.1, label="Progress (%)", interactive=False)
    log_out = gr.Textbox(label="Live logs", lines=16)

    out_json = gr.File(label="Download transcript.json (Format C)")
    out_txt = gr.File(label="Download transcript.txt")
    out_summary = gr.File(label="Download summary.json or summary.md (if enabled)")
    out_log = gr.File(label="Download full session log (.log)")

    run_btn.click(
        fn=run_job_stream,
        inputs=[file_in, whisper_model, device, do_summary, ollama_model, ollama_url, chunk_seconds, episode_override],
        outputs=[progress, log_out, out_json, out_txt, out_summary, out_log],
    )


if __name__ == "__main__":
    acquire_instance_lock()
    apply_runtime_limits()
    demo.queue(default_concurrency_limit=1)
    demo.launch()
