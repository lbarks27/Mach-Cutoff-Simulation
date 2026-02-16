from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer


def _build_pdf_bytes(body_size: float, heading_size: float, title_size: float) -> bytes:
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=title_size,
        leading=title_size + 2,
        textColor=colors.HexColor("#1F2A44"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=body_size,
        leading=body_size + 2,
        textColor=colors.HexColor("#3D4E6C"),
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=heading_size,
        leading=heading_size + 1.5,
        textColor=colors.HexColor("#1F2A44"),
        spaceBefore=4,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=body_size,
        leading=body_size + 2,
        textColor=colors.black,
        spaceAfter=2,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=0,
        spaceAfter=1.5,
    )
    code_style = ParagraphStyle(
        "Code",
        parent=body_style,
        fontName="Courier",
        fontSize=max(7.8, body_size - 0.4),
        leading=max(10.0, body_size + 1.6),
        leftIndent=12,
        textColor=colors.HexColor("#202020"),
    )
    note_style = ParagraphStyle(
        "Note",
        parent=body_style,
        fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#4D4D4D"),
    )

    what_it_is = (
        "Mach Cutoff Simulation is an experimental Python package for simulating "
        "supersonic sonic-boom ray propagation with time-varying HRRR atmosphere data. "
        "It evaluates whether emitted shock rays intersect the ground or cut off along "
        "a waypoint-defined route."
    )

    who_for = (
        "Primary user/persona: Not found in repo. Closest evidence suggests "
        "research and engineering users running sonic-boom propagation experiments."
    )

    feature_bullets = [
        "Loads waypoint JSON (<font name='Courier'>lat/lon/alt/time</font>) and interpolates the flight path over time.",
        "Retrieves HRRR pressure-level snapshots for emission times, with local caching and S3/HTTP retrieval paths.",
        "Computes sound speed, projected wind component, and effective sound speed from atmospheric state.",
        "Builds a point-mass supersonic aircraft model and emits cone-distributed shock directions per emission time.",
        "Ray-traces each emission with adaptive RK4 integration and checks terrain/ground intersections.",
        "Writes summary and hit artifacts, then renders Matplotlib, Plotly, and PyVista outputs.",
    ]

    architecture_bullets = [
        "<font name='Courier'>mach_cutoff/cli.py</font> parses args and loads JSON config via <font name='Courier'>mach_cutoff/config.py</font>.",
        "<font name='Courier'>MachCutoffSimulator</font> in <font name='Courier'>mach_cutoff/simulation/engine.py</font> orchestrates the run loop.",
        "<font name='Courier'>HRRRDatasetManager</font> and <font name='Courier'>HRRRInterpolator</font> fetch and sample atmospheric snapshots.",
        "<font name='Courier'>build_acoustic_grid_field</font> builds refractive-index fields; <font name='Courier'>integrate_ray</font> propagates rays.",
        "<font name='Courier'>SimulationResult</font> serializes JSON/NPZ and feeds visualization backends.",
    ]

    run_steps = [
        "python3 -m venv .venv && source .venv/bin/activate",
        "python3 -m pip install --upgrade pip && python3 -m pip install -e '.[all]'",
        "mach-cutoff --waypoints examples/waypoints_example.json --config examples/config_example.json --output-dir outputs",
    ]

    story = [
        Paragraph("Mach Cutoff Simulation - One-Page App Summary", title_style),
        Paragraph("Evidence source: repository files only (README and mach_cutoff modules).", subtitle_style),
        Paragraph("What It Is", heading_style),
        Paragraph(what_it_is, body_style),
        Paragraph("Who It Is For", heading_style),
        Paragraph(who_for, body_style),
        Paragraph("What It Does", heading_style),
    ]

    story.append(
        ListFlowable(
            [ListItem(Paragraph(text, bullet_style), leftIndent=2) for text in feature_bullets],
            bulletType="bullet",
            leftIndent=10,
            bulletFontName="Helvetica",
            bulletFontSize=max(7.5, body_size),
            spaceBefore=1,
            spaceAfter=2,
        )
    )

    story.extend(
        [
            Paragraph("How It Works (Architecture)", heading_style),
            ListFlowable(
                [ListItem(Paragraph(text, bullet_style), leftIndent=2) for text in architecture_bullets],
                bulletType="bullet",
                leftIndent=10,
                bulletFontName="Helvetica",
                bulletFontSize=max(7.5, body_size),
                spaceBefore=1,
                spaceAfter=2,
            ),
            Paragraph(
                "Data flow: waypoint JSON + config -> simulator -> HRRR snapshots/interpolation -> "
                "acoustic field -> ray integration -> SimulationResult -> JSON/NPZ + visual outputs.",
                note_style,
            ),
            Paragraph("How To Run (Minimal)", heading_style),
        ]
    )

    for cmd in run_steps:
        story.append(Paragraph(cmd, code_style))

    story.append(
        Paragraph(
            "Expected outputs: outputs/simulation_summary.json, outputs/simulation_hits.npz, and enabled backend artifacts.",
            body_style,
        )
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=34,
        rightMargin=34,
        topMargin=30,
        bottomMargin=30,
        title="Mach Cutoff Simulation Summary",
        author="Codex",
    )
    doc.build(story)
    return buffer.getvalue()


def generate(output_path: Path) -> tuple[float, int]:
    # Try progressively denser typography until this fits exactly one page.
    attempts = [
        (9.6, 11.4, 16.5),
        (9.2, 11.0, 16.0),
        (8.9, 10.6, 15.5),
        (8.5, 10.2, 15.0),
    ]
    chosen = attempts[-1]
    pdf_bytes = b""
    pages = 0

    for body, heading, title in attempts:
        candidate = _build_pdf_bytes(body_size=body, heading_size=heading, title_size=title)
        candidate_pages = len(PdfReader(io.BytesIO(candidate)).pages)
        if candidate_pages == 1:
            chosen = (body, heading, title)
            pdf_bytes = candidate
            pages = candidate_pages
            break

    if not pdf_bytes:
        body, heading, title = chosen
        pdf_bytes = _build_pdf_bytes(body_size=body, heading_size=heading, title_size=title)
        pages = len(PdfReader(io.BytesIO(pdf_bytes)).pages)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf_bytes)
    return chosen[0], pages


def main() -> None:
    output_pdf = Path("output/pdf/mach_cutoff_app_summary.pdf")
    chosen_body_size, pages = generate(output_pdf)
    print(f"wrote={output_pdf}")
    print(f"pages={pages}")
    print(f"body_font_size={chosen_body_size}")


if __name__ == "__main__":
    main()
