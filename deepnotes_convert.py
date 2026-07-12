import subprocess
import shutil
import platform
import re
from datetime import datetime
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ClearMetadataPreprocessor


# ---- Configuration ----
SRC = Path(r"C:\Users\pietr\Downloads\Part1 Revised.ipynb")        # the input notebook you want to convert
DEST_DIR = Path(r"C:\Users\pietr\agentchattr-main\SMM921\PART1")       # folder where the final .qmd should end up
KEEP_INTERMEDIATE = False
KEEP_ORIGINAL_NAME = True
RENDER_AFTER = True          # set False to skip rendering
# ------------------------

def strip_deepnote_badge(qmd: Path) -> None:
    """Remove the Deepnote attribution <a>...</a> block if present."""
    text = qmd.read_text()
    # Remove any <a ...deepnote.com...>...</a> block, across newlines
    cleaned = re.sub(
        r"<a\b[^>]*deepnote\.com[^>]*>.*?</a>\s*",
        "",
        text,
        flags=re.S,
    )
    qmd.write_text(cleaned)

def notify(title: str, message: str) -> None:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["osascript", "-e",
                f'display notification "{message}" with title "{title}"'], check=True)
        elif system == "Linux":
            subprocess.run(["notify-send", title, message], check=True)
        elif system == "Windows":
            ps = (f'[reflection.assembly]::loadwithpartialname("System.Windows.Forms");'
                  f'[System.Windows.Forms.MessageBox]::Show("{message}","{title}")')
            subprocess.run(["powershell", "-Command", ps], check=True)
        else:
            raise RuntimeError("unknown platform")
    except Exception:
        print(f"[NOTIFY] {title}: {message}")

def insert_timestamp(qmd: Path, src_name: str, timestamp: str) -> None:
    """Put the timestamp comment just after the YAML front matter."""
    text = qmd.read_text()
    comment = f"<!-- Converted from {src_name} on {timestamp} -->\n"
    # If the file opens with a --- ... --- YAML block, insert after it.
    m = re.match(r"^(---\n.*?\n---\n)", text, re.S)
    if m:
        end = m.end()
        text = text[:end] + "\n" + comment + text[end:]
    else:
        text = comment + "\n" + text
    qmd.write_text(text)

def convert_notebook(src: Path, dest_dir: Path,
                     keep_intermediate: bool = False,
                     keep_original_name: bool = True,
                     render_after: bool = True) -> Path:
    stripped = src.with_name(src.stem + "_clean.ipynb")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Strip metadata
    nb = nbformat.read(str(src), as_version=4)
    nb, _ = ClearMetadataPreprocessor(enabled=True).preprocess(nb, {})
    nbformat.write(nb, str(stripped))

    # 2. Convert to .qmd
    subprocess.run(["quarto", "convert", str(stripped)], check=True)
    produced_qmd = stripped.with_suffix(".qmd")

    # 3. Move to destination first, so render runs in the final location
    final_name = (src.stem if keep_original_name else stripped.stem) + ".qmd"
    final_path = dest_dir / final_name
    shutil.move(str(produced_qmd), str(final_path))

    # 3b. Strip the Deepnote attribution badge
    strip_deepnote_badge(final_path)

    # 4. Timestamp (inserted after YAML front matter)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    insert_timestamp(final_path, src.name, timestamp)

    # 5. Clean up intermediate
    if not keep_intermediate:
        stripped.unlink()

    # 6. Render — produces the output doc next to the .qmd
    if render_after:
        result = subprocess.run(
            ["quarto", "render", final_path.name],
            cwd=str(final_path.parent),     # run inside PART1
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("quarto render failed — see output above")

    return final_path, timestamp

if __name__ == "__main__":
    out, ts = convert_notebook(
        SRC, DEST_DIR,
        keep_intermediate=KEEP_INTERMEDIATE,
        keep_original_name=KEEP_ORIGINAL_NAME,
        render_after=RENDER_AFTER,
    )
    print(f"Wrote and rendered {out}")
    notify("Notebook converted", f"{out.name} converted & rendered ({ts})")
