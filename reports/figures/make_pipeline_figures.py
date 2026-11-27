"""Draw the prompting pipeline figures (main + one per task) as SVG.

No plotting library needed. Render to PNG with any browser, e.g.
msedge --headless --screenshot=out.png --window-size=W,H file.svg
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
COLORS = {  # fill, border
    "data": ("#F1F3F5", "#868E96"),
    "evidence": ("#E7F5FF", "#1C7ED6"),
    "prompt": ("#EDF2FF", "#4263EB"),
    "post": ("#FFF4E6", "#E8590C"),
    "out": ("#EBFBEE", "#2F9E44"),
}
FONT = "Segoe UI, Arial, sans-serif"


def box(x, y, w, h, title, lines=(), kind="data"):
    fill, stroke = COLORS[kind]
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>']
    total = 1 + len(lines)
    top = y + h / 2 - (total - 1) * 13
    parts.append(f'<text x="{x + w / 2}" y="{top + 6}" text-anchor="middle" font-size="21" font-weight="700" fill="#212529">{escape(title)}</text>')
    for i, line in enumerate(lines, 1):
        parts.append(f'<text x="{x + w / 2}" y="{top + 6 + i * 26}" text-anchor="middle" font-size="17" fill="#495057">{escape(line)}</text>')
    return "".join(parts), (x, y, w, h)


def arrow(a, b, label=""):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if abs((ay + ah / 2) - (by + bh / 2)) < 2:  # same row
        x1, y1, x2, y2 = ax + aw, ay + ah / 2, bx, by + bh / 2
        path = f"M{x1},{y1} L{x2 - 4},{y2}"
    else:  # elbow
        x1, y1, x2, y2 = ax + aw, ay + ah / 2, bx, by + bh / 2
        mid = (x1 + x2) / 2
        path = f"M{x1},{y1} L{mid},{y1} L{mid},{y2} L{x2 - 4},{y2}"
    text = (f'<text x="{(x1 + x2) / 2}" y="{min(y1, y2) - 8}" text-anchor="middle" font-size="15" '
            f'fill="#868E96">{escape(label)}</text>') if label else ""
    return f'<path d="{path}" fill="none" stroke="#495057" stroke-width="2.5" marker-end="url(#a)"/>{text}'


def legend(y):
    items = [("evidence", "Bằng chứng"), ("prompt", "Prompt LLM"), ("post", "Hậu xử lý bằng code"), ("out", "File nộp")]
    out, x = [], 40
    for kind, name in items:
        fill, stroke = COLORS[kind]
        out.append(f'<rect x="{x}" y="{y}" width="22" height="22" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        out.append(f'<text x="{x + 30}" y="{y + 17}" font-size="16" fill="#495057">{name}</text>')
        x += 60 + len(name) * 9
    return "".join(out)


def svg(name, width, height, title, body):
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="{FONT}"><defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#495057"/></marker></defs>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="40" y="48" font-size="28" font-weight="700" fill="#212529">{escape(title)}</text>'
        + "".join(body) + legend(height - 42) + "</svg>"
    )
    (OUT / f"{name}.svg").write_text(content, encoding="utf-8")
    print(name, width, height)


def chain(specs, y, h, x0=40, gap=50, width=1720):
    """Boxes in one row; specs = (title, lines, kind). Returns (svg parts, rects)."""
    w = (width - x0 - gap * (len(specs) - 1)) / len(specs)
    parts, rects = [], []
    for i, (title, lines, kind) in enumerate(specs):
        part, rect = box(x0 + i * (w + gap), y, w, h, title, lines, kind)
        parts.append(part)
        rects.append(rect)
    parts += [arrow(rects[i], rects[i + 1]) for i in range(len(rects) - 1)]
    return parts, rects


def main_figure():
    parts, _ = chain([
        ("Input", ["Đề bài", "Code sinh viên"], "data"),
        ("Bằng chứng", ["Biên dịch C++11", "Chạy test do LLM", "sinh từ đề bài"], "evidence"),
        ("RAG", ["BM25 lấy 1 bài", "train giống nhất", "(32 bài có nhãn)"], "evidence"),
        ("Prompt", ["Qwen3.5-9B", "temperature 0"], "prompt"),
        ("Hậu xử lý", ["Luật dựa trên", "bằng chứng", "và định nghĩa"], "post"),
        ("File nộp", ["task1 / 2 / 3", ".json"], "out"),
    ], 110, 170)
    svg("pipeline_main", 1760, 380, "Pipeline chung: Prompt → Hậu xử lý", parts)


def task1_figure():
    parts, _ = chain([
        ("Bằng chứng", ["Compile log + test report", "Bài comment toàn bộ:", "dùng code trong comment"], "evidence"),
        ("Prompt v004", ["Rubric 6 chiều + mốc điểm", "Cách đọc test report", "1 ví dụ RAG"], "prompt"),
        ("Output LLM", ["JSON 6 điểm thành phần", "(không sinh tổng)"], "prompt"),
        ("Hậu xử lý", ["Tổng = cộng 6 chiều", "bằng code"], "post"),
        ("task1.json", ["QWK 0.8865"], "out"),
    ], 110, 170)
    svg("pipeline_task1", 1760, 380, "Task 1: chấm điểm rubric", parts)


def task2_figure():
    W, H, gap = 330, 120, 40
    xs = [40, 40 + W + gap, 40 + 2 * (W + gap)]
    rows = [
        (("Prompt v004", ["Liệt kê nhãn (1 câu/bài)"]), "7 nhãn thường"),
        (("Prompt per_label_v002", ["Yes/No từng nhãn", "+ 2 ví dụ đề khác"]), "3 nhãn hiếm"),
        (("Prompt per_label_v001", ["Yes/No từng nhãn"]), "Lỗi edge case"),
    ]
    parts, ends = [], []
    for i, ((title, lines), pick) in enumerate(rows):
        y = 100 + i * (H + 35)
        p, r1 = box(xs[0], y, W, H, title, lines, "prompt")
        q, r2 = box(xs[1], y, W, H, "Luật bằng chứng", ["Biên dịch lỗi ⇄ Lỗi biên dịch", "Qua hết test: bỏ nhãn test thấy"], "post")
        s, r3 = box(xs[2], y, W, H, "Chọn", [pick], "post")
        parts += [p, q, s, arrow(r1, r2), arrow(r2, r3)]
        ends.append(r3)
    edge_y = 100 + 3 * (H + 35)
    e, re_ = box(xs[2], edge_y, W, H, "Luật edge case", ["Test thường qua hết", "nhưng test biên rớt"], "post")
    parts.append(e)
    ends.append(re_)
    m, rm = box(xs[2] + W + 70, 100 + 1.5 * (H + 35) - 10, 300, H + 20, "Ghép nhãn", ["thường + hiếm", "+ edge case (Yes hoặc luật)"], "post")
    o, ro = box(xs[2] + W + 70 + 300 + 50, 100 + 1.5 * (H + 35) - 10, 210, H + 20, "task2.json", ["macro-F1 0.3027"], "out")
    parts += [m, o, arrow(rm, ro)] + [arrow(r, rm) for r in ends]
    width = int(xs[2] + W + 70 + 300 + 50 + 210 + 40)
    svg("pipeline_task2", width, edge_y + H + 90, "Task 2: phân loại lỗi (3 cách hỏi rồi ghép)", parts)


def task3_figure():
    parts, _ = chain([
        ("Input", ["Nhãn lỗi từ task2.json", "Level mục tiêu", "+ bằng chứng"], "data"),
        ("Prompt v008", ["Viết feedback đúng", "định nghĩa level", "khớp nhãn lỗi"], "prompt"),
        ("Kiểm tra level", ["Vi phạm (đưa code,", "lời giải...) → sinh lại", "tối đa 2 lần"], "post"),
        ("Hậu xử lý", ["L1: 1 trong 2 câu chuẩn", "L2: cắt câu khuyên sửa", "Nhãn = Task 2"], "post"),
        ("task3.json", ["LLM judge 0.4223"], "out"),
    ], 110, 170)
    svg("pipeline_task3", 1760, 380, "Task 3: viết feedback theo level", parts)


if __name__ == "__main__":
    main_figure()
    task1_figure()
    task2_figure()
    task3_figure()
