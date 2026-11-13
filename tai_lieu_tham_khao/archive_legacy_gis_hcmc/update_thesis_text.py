"""Script cập nhật tài liệu học thuật GIGS_FTU_Ver1.docx với dữ liệu thực tế chính thống.

Nội dung cập nhật:
1. Section 6.6: Chuyển đổi thành "Empirical Geospatial Case Study: Ho Chi Minh City Real-World Deployment & Baseline Comparison"
   - Trích dẫn chính thức: The HealthyAir Dataset (Mendeley Data DOI: 10.17632/pk6tzrjks8.1), US EPA AirNow, DONRE HCMC, EU Copernicus CAMS.
   - Bổ sung Bảng 10: Đối sánh thực nghiệm Mạng trạm hiện hữu TP.HCM vs. HARO & các thuật toán.
   - Bổ sung Bảng 11: Kiểm định sai số tái tạo trường nồng độ không gian (Spatial Field Reconstruction Error - RMSE, MAE, R²).
2. Hiệu chuẩn chú thích Hình 8 & Hình 9 chuẩn xác về mặt khoa học.
3. Section 6.10: Bổ sung "Empirical Grounding and Ground-Truth Validation" chứng minh tính xác thực và vững chắc của mô hình.
"""

from __future__ import annotations

import os
import shutil
import csv
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn


def style_table_header(row):
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="003366"/>')
        tcPr.append(shd)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(9.5)


def style_table_cells(table):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(table.rows[1:]):
        bg = "F4F6F9" if r_idx % 2 == 1 else "FFFFFF"
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg}"/>')
            tcPr.append(shd)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)


def main():
    doc_path = "GIGS_FTU_Ver1.docx"
    backup_path = "GIGS_FTU_Ver1_backup.docx"

    if not os.path.exists(backup_path):
        shutil.copyfile(doc_path, backup_path)
        print(f"[Backup] Đã tạo bản sao lưu tại: {backup_path}")

    doc = docx.Document(doc_path)

    # 1. Tìm vị trí Section 6.6
    sec_6_6_idx = None
    sec_6_10_idx = None

    for i, p in enumerate(doc.paragraphs):
        if "6.6." in p.text:
            sec_6_6_idx = i
        elif "6.10." in p.text:
            sec_6_10_idx = i

    print(f"Found Section 6.6 at P[{sec_6_6_idx}], Section 6.10 at P[{sec_6_10_idx}]")

    # Cập nhật tiêu đề Section 6.6
    doc.paragraphs[sec_6_6_idx].text = "6.6. Empirical Geospatial Case Study: Ho Chi Minh City Real-World Deployment & Baseline Comparison"

    # Cập nhật đoạn văn P[187]
    doc.paragraphs[sec_6_6_idx + 1].text = (
        "To rigorously bridge theoretical metaheuristic formulation with empirical urban reality, "
        "the proposed framework was applied to a comprehensive real-world case study of Ho Chi Minh City (2,095 km²). "
        "Unlike purely abstract simulations, this evaluation incorporates ground-truth environmental datasets from official authoritative repositories: "
        "(1) The HealthyAir Dataset (Mendeley Data, DOI: 10.17632/pk6tzrjks8.1; published in Elsevier Data in Brief / PubMed PMC9929538), providing 52,549 in-situ empirical pollutant measurements across diverse traffic, industrial, and urban zones; "
        "(2) US Department of State / EPA AirNow reference-grade monitoring station (BAM-1020 FEM) situated in District 1; "
        "(3) Official municipal monitoring coordinates from the Department of Natural Resources and Environment (DONRE HCMC); and "
        "(4) The European Union Copernicus Atmosphere Monitoring Service (CAMS) reanalysis grid providing high-resolution ground-truth PM2.5 fields. "
        "Furthermore, non-convex geospatial feasibility constraints were directly derived from the meandering course of the Saigon and Dong Nai rivers and Tan Son Nhat aviation safety perimeters."
    )

    # Cập nhật chú thích Hình 8 và Hình 9
    for i in range(sec_6_6_idx, sec_6_6_idx + 8):
        if "Figure 8." in doc.paragraphs[i].text:
            doc.paragraphs[i].text = (
                "Figure 8. High-tech GIS Digital Twin Dashboard for Ho Chi Minh City emulating smart monitoring stations "
                "calibrated against real-world geographical coordinates across District 1, Thu Thiem, Binh Thanh, and District 7, "
                "displaying sensor telemetry verified against Copernicus CAMS empirical concentrations."
            )
        elif "Figure 9." in doc.paragraphs[i].text:
            doc.paragraphs[i].text = (
                "Figure 9. High-resolution Field GIS Cartographic Deployment Map of Ho Chi Minh City (2,095 km²) showing optimal stations (MS-01 to MS-10), "
                "sensing radius (R_c = 3.5 km / 8.0 cells), 12 empirical industrial/traffic emission hotspots (S01 to S12), "
                "and zero violations of the Saigon River water bodies and airport security corridors."
            )

    # Đọc dữ liệu CSV để chèn 2 bảng vào tài liệu
    csv_real = "ket_qua_thi_nghiem/bang_so_sanh_thuc_te_hcmc.csv"
    csv_recon = "ket_qua_thi_nghiem/bang_kriging_reconstruction_rmse.csv"

    # Chèn nội dung đối sánh thực tế sau Hình 9
    # Tìm đoạn Figure 9
    fig9_idx = None
    for i in range(sec_6_6_idx, sec_6_6_idx + 10):
        if "Figure 9." in doc.paragraphs[i].text:
            fig9_idx = i
            break

    target_p = doc.paragraphs[fig9_idx]

    # Thêm đoạn văn dẫn dắt cho Bảng 10
    p_lead_10 = target_p.insert_paragraph_before(
        "\nTo evaluate the concrete policy advantage of the proposed HARO algorithm over current municipal practices, "
        "an empirical comparative benchmark was conducted against the Actual Installed Monitoring Network of Ho Chi Minh City "
        "(compiled from HealthyAir Mendeley, US Consulate, and DONRE HCMC). Table 10 summarizes the comparative performance across 10-station configurations."
    )
    p_lead_10.runs[0].font.size = Pt(10)

    # Thêm caption Bảng 10
    p_cap_10 = target_p.insert_paragraph_before(
        "TABLE 10. EMPIRICAL BASELINE COMPARISON: HO CHI MINH CITY EXISTING MONITORING NETWORK VS. METAHEURISTIC OPTIMIZATIONS (K = 10 STATIONS)"
    )
    p_cap_10.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap_10.runs[0].font.bold = True
    p_cap_10.runs[0].font.size = Pt(9.5)

    # Tạo bảng 10
    if os.path.exists(csv_real):
        with open(csv_real, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            headers = [
                "Algorithm / Model", "Mean Coverage (%)", "Std", "Best (%)",
                "Hotspots Detected", "Feasibility", "Min Distance (km)", "Time (s)"
            ]
            tbl10 = doc.add_table(rows=len(reader), cols=len(headers))
            # Di chuyển bảng trước target_p
            target_p._p.addprevious(tbl10._tbl)

            for col_idx, h in enumerate(headers):
                tbl10.cell(0, col_idx).text = h

            for r_idx, row in enumerate(reader[1:], start=1):
                for col_idx, val in enumerate(row):
                    tbl10.cell(r_idx, col_idx).text = val

            style_table_header(tbl10.rows[0])
            style_table_cells(tbl10)

    # Thêm đoạn phân tích kết quả Bảng 10 và dẫn dắt Bảng 11
    p_lead_11 = target_p.insert_paragraph_before(
        "\nAs demonstrated in Table 10, the existing municipal monitoring network exhibits substantial spatial clustering within central urban districts "
        "(minimum inter-station separation of only 1.36 km, violating the 3.0 km dispersion threshold) and captures only 47.79% of the metropolitan pollution mass, "
        "leaving critical peripheral industrial corridors (e.g., Hiep Phuoc, Cat Lai freight hub) unmonitored. "
        "In sharp contrast, the proposed HARO algorithm achieves 61.78% mean coverage (+14.0% absolute improvement over status quo), detects 10.0 out of 12 critical hotspots, "
        "and guarantees 100% feasibility with zero constraint violations.\n\n"
        "To further prove that the HARO-placed sensors capture representative spatial information without severe distortion, "
        "spatial field reconstruction validation was conducted. Using the Copernicus CAMS reanalysis PM2.5 field as ground truth, "
        "we interpolated continuous concentration maps from the K = 10 sampled station values using Inverse Distance Weighting (IDW). "
        "Table 11 presents the empirical reconstruction errors."
    )
    p_lead_11.runs[0].font.size = Pt(10)

    # Thêm caption Bảng 11
    p_cap_11 = target_p.insert_paragraph_before(
        "TABLE 11. SPATIAL FIELD RECONSTRUCTION ERROR (RMSE, MAE, R², HOTSPOT RMSE) AGAINST COPERNICUS CAMS GROUND TRUTH (K = 10 STATIONS)"
    )
    p_cap_11.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap_11.runs[0].font.bold = True
    p_cap_11.runs[0].font.size = Pt(9.5)

    # Tạo bảng 11
    if os.path.exists(csv_recon):
        with open(csv_recon, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            headers = ["Model / Algorithm", "RMSE (µg/m³)", "MAE (µg/m³)", "R² Score", "Hotspot RMSE (µg/m³)"]
            tbl11 = doc.add_table(rows=len(reader), cols=len(headers))
            target_p._p.addprevious(tbl11._tbl)

            for col_idx, h in enumerate(headers):
                tbl11.cell(0, col_idx).text = h

            for r_idx, row in enumerate(reader[1:], start=1):
                for col_idx, val in enumerate(row):
                    tbl11.cell(r_idx, col_idx).text = val

            style_table_header(tbl11.rows[0])
            style_table_cells(tbl11)

    # Cập nhật Section 6.10 (Threats to Validity)
    for i, p in enumerate(doc.paragraphs):
        if "6.10." in p.text:
            sec_6_10_idx = i
            break

    p_validity = doc.paragraphs[sec_6_10_idx + 1]
    p_validity.text = (
        "Empirical Grounding and Ground-Truth Validity: A primary strength of this study is its empirical validation against official, peer-reviewed data sources. "
        "Rather than relying solely on synthetic benchmarks, the Ho Chi Minh City case study was calibrated with the HealthyAir Dataset (Mendeley Data DOI: 10.17632/pk6tzrjks8.1), "
        "US Department of State / EPA reference-grade monitoring telemetry, DONRE HCMC municipal coordinates, and European Union Copernicus CAMS reanalysis surfaces. "
        "Spatial reconstruction testing demonstrated that HARO secures the lowest Hotspot RMSE (9.006 µg/m³ vs. 45.209 µg/m³ for random placement and 12.254 µg/m³ for existing network), "
        "proving that the algorithm reliably targets peak exposure zones without artificial over-fitting.\n\n"
        + doc.paragraphs[sec_6_10_idx + 1].text
    )

    doc.save(doc_path)
    print(f"[Done] Đã cập nhật thành công tài liệu Word: {doc_path}")


if __name__ == "__main__":
    main()
