"""
DeepForensics PDF Report Generator (Module 8).

Produces a multi-page forensic PDF from an AnalysisResult using
ReportLab.  The report consolidates the verdict, semantic evidence,
Grad-CAM heatmap, and audit block into a single archival document.

Page layout:
  Page 1 — Verdict, confidence, and natural-language summary
  Page 2 — Semantic score panels (lip-sync, blink, lighting)
  Page 3 — Grad-CAM heatmap (full-size)
  Page 4 — Audit block (checksums, timestamps, per-module times)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from deepforensics.result import AnalysisResult

logger = logging.getLogger(__name__)


def generate_report(result: AnalysisResult, output_path: str) -> str:
    """
    Generate a forensic PDF report from an AnalysisResult.

    Parameters
    ----------
    result : AnalysisResult
        The complete analysis result.
    output_path : str
        Path where the PDF will be saved.

    Returns
    -------
    str
        The absolute path to the generated PDF.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install reportlab"
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # -- Custom styles ------------------------------------------------------
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        spaceAfter=12,
        textColor=colors.HexColor("#1a1a2e"),
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=16,
        textColor=colors.HexColor("#16213e"),
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    verdict_style = ParagraphStyle(
        "Verdict",
        parent=styles["Heading1"],
        fontSize=24,
        alignment=1,  # centre
        spaceAfter=8,
        textColor=(
            colors.HexColor("#c0392b")
            if result.verdict == "FAKE"
            else colors.HexColor("#27ae60")
        ),
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
    )

    # ======================================================================
    # PAGE 1 — Verdict & Summary
    # ======================================================================
    elements.append(Paragraph("DeepForensics — Forensic Analysis Report", title_style))
    elements.append(Spacer(1, 6 * mm))

    # Input metadata table
    meta_data = [
        ["File", result.input_filename or "Unknown"],
        ["Media Type", result.input_media_type or "Unknown"],
        ["Frames Processed", str(result.num_frames_processed or "N/A")],
        ["Duration", f"{result.duration_seconds:.1f}s" if result.duration_seconds else "N/A"],
        ["Analysis Time", f"{result.audit.total_time:.1f}s"],
        ["Timestamp", result.audit.timestamp],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 12 * cm])
    meta_table.setStyle(
        TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 10 * mm))

    # Verdict
    elements.append(Paragraph(result.verdict, verdict_style))
    conf_pct = f"{result.confidence * 100:.1f}%"
    elements.append(
        Paragraph(f"Confidence: {conf_pct}", ParagraphStyle(
            "ConfCenter", parent=body_style, alignment=1, fontSize=14
        ))
    )
    elements.append(Spacer(1, 8 * mm))

    # Summary
    elements.append(Paragraph("Evidence Summary", heading_style))
    elements.append(Paragraph(result.summary, body_style))

    if result.visual_only_confidence is not None:
        elements.append(Spacer(1, 4 * mm))
        elements.append(
            Paragraph(
                f"Visual-only baseline confidence: "
                f"{result.visual_only_confidence * 100:.1f}%",
                small_style,
            )
        )

    elements.append(PageBreak())

    # ======================================================================
    # PAGE 2 — Semantic Scores
    # ======================================================================
    elements.append(Paragraph("Semantic Analysis Scores", title_style))
    elements.append(Spacer(1, 6 * mm))

    # Lip-sync
    elements.append(Paragraph("Lip-Audio Coherence (Module 4A)", heading_style))
    if result.semantic_scores.lip_sync:
        ls = result.semantic_scores.lip_sync
        ls_data = [
            ["Metric", "Value"],
            ["Mean Correlation (r)", f"{ls.mean_correlation:.4f}"],
            ["Min Correlation", f"{ls.min_correlation:.4f}"],
            ["Std Deviation", f"{ls.std_correlation:.4f}"],
            ["Below-Threshold Fraction", f"{ls.below_threshold_fraction:.2%}"],
        ]
        ls_table = Table(ls_data, colWidths=[6 * cm, 5 * cm])
        ls_table.setStyle(_score_table_style())
        elements.append(ls_table)
    else:
        elements.append(Paragraph("Audio unavailable — lip-sync analysis skipped.", body_style))

    elements.append(Spacer(1, 6 * mm))

    # Blink
    elements.append(Paragraph("Eye-Blink Dynamics (Module 4B)", heading_style))
    if result.semantic_scores.blink:
        bk = result.semantic_scores.blink
        bk_data = [
            ["Metric", "Value"],
            ["Blink Rate", f"{bk.blink_rate_per_min:.1f} / min"],
            ["Mean Duration", f"{bk.mean_duration_frames:.1f} frames"],
            ["IBI Std Dev", f"{bk.ibi_std:.4f} s"],
            ["Anomaly Z-Score", f"{bk.anomaly_z_score:.2f}"],
        ]
        bk_table = Table(bk_data, colWidths=[6 * cm, 5 * cm])
        bk_table.setStyle(_score_table_style())
        elements.append(bk_table)
    else:
        elements.append(Paragraph("Single image input — blink analysis skipped.", body_style))

    elements.append(Spacer(1, 6 * mm))

    # Lighting
    elements.append(Paragraph("Lighting Consistency (Module 4C)", heading_style))
    if result.semantic_scores.lighting:
        lt = result.semantic_scores.lighting
        lt_data = [
            ["Metric", "Value"],
            ["Direction Variance", f"{lt.direction_variance:.6f}"],
            ["Max Pairwise Angle", f"{lt.max_pairwise_angle_deg:.1f}°"],
            ["Mean Intensity Ratio", f"{lt.mean_intensity_ratio:.4f}"],
            ["Consistency Score", f"{lt.consistency_score:.4f}"],
        ]
        lt_table = Table(lt_data, colWidths=[6 * cm, 5 * cm])
        lt_table.setStyle(_score_table_style())
        elements.append(lt_table)
    else:
        elements.append(Paragraph("Lighting analysis unavailable.", body_style))

    elements.append(PageBreak())

    # ======================================================================
    # PAGE 3 — Grad-CAM Heatmap
    # ======================================================================
    elements.append(Paragraph("Grad-CAM Heatmap (Module 7)", title_style))
    elements.append(Spacer(1, 6 * mm))

    if result.gradcam.overlay_path and os.path.exists(result.gradcam.overlay_path):
        elements.append(
            Image(result.gradcam.overlay_path, width=12 * cm, height=12 * cm)
        )
        elements.append(Spacer(1, 4 * mm))
        if result.gradcam.region_label:
            elements.append(
                Paragraph(
                    f"Dominant region: <b>{result.gradcam.region_label}</b>",
                    body_style,
                )
            )
        if result.gradcam.peak_coordinate:
            elements.append(
                Paragraph(
                    f"Peak activation: ({result.gradcam.peak_coordinate[0]}, "
                    f"{result.gradcam.peak_coordinate[1]})",
                    small_style,
                )
            )
    elif result.gradcam.heatmap_path and os.path.exists(result.gradcam.heatmap_path):
        elements.append(
            Image(result.gradcam.heatmap_path, width=12 * cm, height=12 * cm)
        )
    else:
        elements.append(Paragraph("Grad-CAM heatmap not available.", body_style))

    elements.append(PageBreak())

    # ======================================================================
    # PAGE 4 — Audit Block
    # ======================================================================
    elements.append(Paragraph("Audit & Reproducibility Record", title_style))
    elements.append(Spacer(1, 6 * mm))

    audit = result.audit
    audit_data = [
        ["Field", "Value"],
        ["Timestamp", audit.timestamp],
        ["Input File SHA-256", audit.input_file_sha256 or "N/A"],
        ["Backbone Checkpoint SHA-256", audit.backbone_checkpoint_sha256 or "N/A"],
        ["Fusion Checkpoint SHA-256", audit.fusion_checkpoint_sha256 or "N/A"],
        ["Total Inference Time", f"{audit.total_time:.3f} s"],
    ]
    audit_table = Table(audit_data, colWidths=[6 * cm, 10 * cm])
    audit_table.setStyle(_score_table_style())
    elements.append(audit_table)

    elements.append(Spacer(1, 8 * mm))

    # Per-module timing breakdown
    elements.append(Paragraph("Per-Module Timing", heading_style))
    if audit.module_times:
        time_data = [["Module", "Time (s)"]]
        for module_name, elapsed in audit.module_times.items():
            time_data.append([module_name.capitalize(), f"{elapsed:.3f}"])
        time_table = Table(time_data, colWidths=[6 * cm, 5 * cm])
        time_table.setStyle(_score_table_style())
        elements.append(time_table)

    # Footer note
    elements.append(Spacer(1, 12 * mm))
    elements.append(
        Paragraph(
            "This report was generated by DeepForensics — an explainable "
            "forensic framework for deepfake detection. All numbers are "
            "computed from the pixels and the waveform by committed, "
            "hashed, and reproducible model weights.",
            small_style,
        )
    )

    # -- Build PDF ----------------------------------------------------------
    doc.build(elements)
    logger.info("Forensic report saved to %s", output_path)
    return os.path.abspath(output_path)


def _score_table_style():
    """Shared table style for score panels."""
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle

    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
    ])
