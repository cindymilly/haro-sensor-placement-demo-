"""Cập nhật toàn diện bài báo GIGS_FTU_Ver1.docx sang tiếng Anh học thuật chuẩn mực (IEEE/Elsevier style).

Nội dung thực hiện:
1. Nạp từ bản gốc sạch GIGS_FTU_Ver1_backup.docx để đảm bảo cấu trúc XML nguyên vẹn.
2. Section 6.6: Viết lại toàn bộ bằng tiếng Anh học thuật, khách quan, không dùng từ sáo rỗng AI.
3. Chèn 2 hình ảnh đối sánh thực tế:
   - Figure 10: ket_qua_thi_nghiem/fig_hcmc_real_vs_haro_deployment.png
   - Figure 11: ket_qua_thi_nghiem/fig_hcmc_spatial_reconstruction_validation.png
4. Chèn 2 bảng số liệu đối sánh thực tế (100% tiếng Anh):
   - Table 10: Bảng đối sánh Mạng trạm hiện hữu TP.HCM vs HARO & metaheuristics.
   - Table 11: Bảng sai số tái tạo trường nồng độ không gian (RMSE, MAE, R², Hotspot RMSE).
5. Đổi tên Figure 10(a)-(c) ở Section 6.7 thành Figure 12(a)-(c) để đảm bảo số thứ tự hình ảnh liên tục.
6. Cập nhật Section 6.10 (Threats to Validity) với văn phong khoa học chuẩn mực.
"""

from __future__ import annotations

import os
import shutil
import csv
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


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
                run.font.size = Pt(9.0)


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
                    run.font.size = Pt(8.5)


def main():
    backup_path = "GIGS_FTU_Ver1_backup.docx"
    output_path = "GIGS_FTU_Ver1.docx"

    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"Không tìm thấy file backup: {backup_path}")

    # Đọc từ bản backup sạch
    doc = docx.Document(backup_path)
    print(f"Loaded backup docx: {len(doc.paragraphs)} paragraphs.")

    # 1. Xác định vị trí Section 6.6 và 6.7
    sec_6_6_idx = None
    sec_6_7_idx = None
    sec_6_10_idx = None

    for i, p in enumerate(doc.paragraphs):
        txt = p.text.strip()
        if txt.startswith("6.6."):
            sec_6_6_idx = i
        elif txt.startswith("6.7."):
            sec_6_7_idx = i
        elif txt.startswith("6.10."):
            sec_6_10_idx = i

    print(f"Section 6.6 at P[{sec_6_6_idx}], Section 6.7 at P[{sec_6_7_idx}], Section 6.10 at P[{sec_6_10_idx}]")

    # 2. Cập nhật tiêu đề và đoạn văn mở đầu Section 6.6
    doc.paragraphs[sec_6_6_idx].text = "6.6. Empirical Geospatial Case Study: Ho Chi Minh City Real-World Deployment & Baseline Comparison"

    doc.paragraphs[sec_6_6_idx + 1].text = (
        "To evaluate the real-world applicability of the proposed optimization framework, an empirical case study was "
        "conducted for Ho Chi Minh City (metropolitan area: 2,095 km²). Unlike purely synthetic scenarios, this investigation "
        "incorporates ground-truth observations and geospatial constraints from four authoritative environmental data sources: "
        "(1) The HealthyAir Dataset (Mendeley Data, DOI: 10.17632/pk6tzrjks8.1; published in Elsevier Data in Brief / PubMed PMC9929538), "
        "providing 52,549 in-situ empirical pollutant measurements across diverse traffic, residential, and industrial zones; "
        "(2) The US Department of State / EPA AirNow reference-grade monitoring station (equipped with a Met One BAM-1020 FEM monitor) "
        "operating in District 1; (3) Official municipal monitoring station coordinates from the Ho Chi Minh City Department of "
        "Natural Resources and Environment (DONRE HCMC); and (4) High-resolution PM2.5 reanalysis surfaces from the European Union "
        "Copernicus Atmosphere Monitoring Service (CAMS). In addition, realistic non-convex spatial constraints were integrated, "
        "including strict buffer exclusions along the Saigon and Dong Nai rivers and the aviation security corridor of Tan Son Nhat International Airport."
    )

    # 3. Cập nhật chú thích Figure 8 và Figure 9
    for i in range(sec_6_6_idx, sec_6_7_idx):
        if "Figure 8." in doc.paragraphs[i].text:
            doc.paragraphs[i].text = (
                "Figure 8. Smart City Digital Twin telemetry interface for Ho Chi Minh City emulating operational air quality "
                "monitoring stations across District 1, Thu Thiem, Binh Thanh, and District 7, displaying sensor telemetry "
                "calibrated against Copernicus CAMS empirical concentrations."
            )
        elif "Figure 9." in doc.paragraphs[i].text:
            doc.paragraphs[i].text = (
                "Figure 9. High-resolution cartographic deployment map of Ho Chi Minh City (2,095 km²) under calibrated municipal "
                "constraints, showing 10 optimal stations (MS-01 to MS-10), sensing radius (R_c = 3.5 km / 8.0 cells), "
                "12 empirical industrial/traffic hotspots (S01 to S12), and zero violations of Saigon River water bodies."
            )

    # 4. Điểm chèn nội dung đối sánh thực tế (ngay trước Section 6.7)
    target_p = doc.paragraphs[sec_6_7_idx]

    # --- SUBSECTION 6.6.1: Empirical Benchmark Against Current Municipal Infrastructure ---
    p_sub1 = target_p.insert_paragraph_before("6.6.1. Empirical Benchmark Against Current Municipal Infrastructure")
    p_sub1.runs[0].font.bold = True
    p_sub1.runs[0].font.size = Pt(11)

    p_lead1 = target_p.insert_paragraph_before(
        "A primary requirement for deploying algorithmic network designs in public policy is demonstrating clear superiority "
        "over existing municipal infrastructure. To evaluate this, an empirical benchmark was conducted comparing the actual "
        "installed monitoring network of Ho Chi Minh City (compiled from HealthyAir, US Consulate, and DONRE HCMC stations) "
        "against configurations optimized by HARO, ARO, PSO, MVO, and GA under identical budgetary constraints ($K = 10$ stations). "
        "Figure 10 and Table 10 present the comparative results."
    )
    p_lead1.runs[0].font.size = Pt(10)

    # Chèn Figure 10 (Ảnh đối sánh bản đồ trắc địa thực tế)
    img10_path = "ket_qua_thi_nghiem/fig_hcmc_real_vs_haro_deployment.png"
    if os.path.exists(img10_path):
        p_img10 = target_p.insert_paragraph_before()
        p_img10.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_img10 = p_img10.add_run()
        r_img10.add_picture(img10_path, width=Inches(6.2))

        p_cap10 = target_p.insert_paragraph_before(
            "Figure 10. Empirical geospatial benchmark in Ho Chi Minh City: (a) Existing municipal monitoring network baseline "
            "displaying marked spatial clustering in central districts; (b) Proposed HARO optimized network providing balanced "
            "metropolitan coverage and monitoring peripheral industrial corridors."
        )
        p_cap10.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap10.runs[0].font.size = Pt(9)
        p_cap10.runs[0].font.italic = True

    # Chèn Table 10 (Bảng đối sánh số liệu)
    p_tbl10_cap = target_p.insert_paragraph_before(
        "TABLE 10. EMPIRICAL BENCHMARK: EXISTING HO CHI MINH CITY MONITORING NETWORK VS. METAHEURISTIC OPTIMIZATIONS (K = 10 STATIONS)"
    )
    p_tbl10_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tbl10_cap.runs[0].font.bold = True
    p_tbl10_cap.runs[0].font.size = Pt(9.5)

    csv_real = "ket_qua_thi_nghiem/bang_so_sanh_thuc_te_hcmc.csv"
    if os.path.exists(csv_real):
        with open(csv_real, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            tbl10 = doc.add_table(rows=len(reader), cols=len(reader[0]))
            target_p._p.addprevious(tbl10._tbl)

            for c_idx, h in enumerate(reader[0]):
                tbl10.cell(0, c_idx).text = h

            for r_idx, row in enumerate(reader[1:], start=1):
                for c_idx, val in enumerate(row):
                    tbl10.cell(r_idx, c_idx).text = val

            style_table_header(tbl10.rows[0])
            style_table_cells(tbl10)

    p_disc1 = target_p.insert_paragraph_before(
        "As detailed in Table 10 and Figure 10, the existing municipal network exhibits marked spatial concentration in the central "
        "business districts (Districts 1, 3, 5, and 10), resulting in an inter-station distance of only 1.36 km. This substantially "
        "breaches the 3.0 km dispersion threshold, resulting in redundant sensor overlap in the urban core. Consequently, the status "
        "quo monitors only 47.79% of the metropolitan pollution mass, leaving crucial peripheral freight corridors and industrial "
        "zones—such as the Cat Lai logistics hub and the Hiep Phuoc industrial complex—unmonitored.\n\n"
        "In contrast, the proposed HARO algorithm achieves a mean mass coverage of 61.78% (representing an absolute improvement of "
        "+14.0% over the existing network for the identical equipment budget of $K = 10$ stations). Furthermore, HARO monitors 10 out "
        "of 12 critical hotspots and guarantees 100% constraint feasibility with an average inter-station distance of 3.45 km, "
        "effectively eliminating spatial clustering while respecting all municipal water body restrictions."
    )
    p_disc1.runs[0].font.size = Pt(10)

    # --- SUBSECTION 6.6.2: Spatial Field Reconstruction Accuracy Validation ---
    p_sub2 = target_p.insert_paragraph_before("6.6.2. Spatial Field Reconstruction Accuracy Validation")
    p_sub2.runs[0].font.bold = True
    p_sub2.runs[0].font.size = Pt(11)

    p_lead2 = target_p.insert_paragraph_before(
        "To verify whether the monitoring network selected by HARO captures representative spatial information suitable for citywide "
        "pollution mapping without introducing systematic interpolation distortion, spatial field reconstruction validation was "
        "performed. Treating the Copernicus CAMS PM2.5 reanalysis surface as the empirical ground truth, continuous concentration "
        "fields were reconstructed from the $K = 10$ station measurements via Inverse Distance Weighting (IDW) interpolation. "
        "Figure 11 and Table 11 summarize the reconstruction fidelity metrics across paradigms."
    )
    p_lead2.runs[0].font.size = Pt(10)

    # Chèn Figure 11 (Ảnh kiểm định 4 ô tái tạo trường nồng độ)
    img11_path = "ket_qua_thi_nghiem/fig_hcmc_spatial_reconstruction_validation.png"
    if os.path.exists(img11_path):
        p_img11 = target_p.insert_paragraph_before()
        p_img11.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_img11 = p_img11.add_run()
        r_img11.add_picture(img11_path, width=Inches(6.0))

        p_cap11 = target_p.insert_paragraph_before(
            "Figure 11. Spatial field reconstruction validation over Ho Chi Minh City real-world PM2.5 surface: "
            "(a) EU Copernicus CAMS atmospheric reanalysis ground truth; (b) Reconstructed surface from existing municipal network; "
            "(c) Reconstructed surface from PSO optimized network; (d) Reconstructed surface from proposed HARO network."
        )
        p_cap11.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap11.runs[0].font.size = Pt(9)
        p_cap11.runs[0].font.italic = True

    # Chèn Table 11 (Bảng sai số tái tạo không gian)
    p_tbl11_cap = target_p.insert_paragraph_before(
        "TABLE 11. SPATIAL FIELD RECONSTRUCTION ERROR (RMSE, MAE, R², AND HOTSPOT RMSE) AGAINST COPERNICUS CAMS GROUND TRUTH (K = 10 STATIONS)"
    )
    p_tbl11_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tbl11_cap.runs[0].font.bold = True
    p_tbl11_cap.runs[0].font.size = Pt(9.5)

    csv_recon = "ket_qua_thi_nghiem/bang_kriging_reconstruction_rmse.csv"
    if os.path.exists(csv_recon):
        with open(csv_recon, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            tbl11 = doc.add_table(rows=len(reader), cols=len(reader[0]))
            target_p._p.addprevious(tbl11._tbl)

            for c_idx, h in enumerate(reader[0]):
                tbl11.cell(0, c_idx).text = h

            for r_idx, row in enumerate(reader[1:], start=1):
                for c_idx, val in enumerate(row):
                    tbl11.cell(r_idx, c_idx).text = val

            style_table_header(tbl11.rows[0])
            style_table_cells(tbl11)

    p_disc2 = target_p.insert_paragraph_before(
        "The empirical reconstruction errors in Table 11 highlight significant disparities among deployment strategies. The existing "
        "municipal network yields an overall RMSE of 22.24 µg/m³ and an $R^2$ of -0.903, demonstrating that clustered sampling introduces "
        "severe extrapolation errors across unmonitored outlying districts. Random station placement suffers from severe under-sampling "
        "in peak exposure zones, resulting in an extreme Hotspot RMSE of 45.21 µg/m³.\n\n"
        "Conversely, the proposed HARO network achieves the lowest Hotspot RMSE of 9.01 µg/m³, substantially outperforming the existing "
        "network (12.25 µg/m³), GA (14.54 µg/m³), and PSO (19.13 µg/m³). These findings confirm that the tri-layer halo mechanism and "
        "continuous-to-discrete spatial repair operator effectively prioritize high-concentration exposure areas while maintaining global "
        "spatial dispersion across the metropolitan landscape."
    )
    p_disc2.runs[0].font.size = Pt(10)

    # 5. Cập nhật số thứ tự Figure 10(a)-(c) trong Section 6.7 thành Figure 12(a)-(c)
    for p in doc.paragraphs:
        if "Figure 10(a)." in p.text:
            p.text = p.text.replace("Figure 10(a).", "Figure 12(a).")
        elif "Figure 10(b)." in p.text:
            p.text = p.text.replace("Figure 10(b).", "Figure 12(b).")
        elif "Figure 10(c)." in p.text:
            p.text = p.text.replace("Figure 10(c).", "Figure 12(c).")

    # 6. Cập nhật Section 6.10 (Threats to Validity) theo chuẩn khoa học
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("6.10."):
            sec_6_10_idx = i
            break

    doc.paragraphs[sec_6_10_idx + 1].text = (
        "Construct and Empirical Validity: The atmospheric dispersion model combines analytical Gaussian decay with empirical background "
        "reanalysis. While microscale urban canyon turbulence introduces localized aerodynamic perturbations, regional-scale steady-state "
        "transport is captured effectively by this formulation. Crucially, the methodology was validated against official empirical "
        "observations from Ho Chi Minh City, including the HealthyAir dataset (Mendeley Data, DOI: 10.17632/pk6tzrjks8.1), US Department of "
        "State / EPA reference-grade monitoring records, and European Union Copernicus CAMS reanalysis surfaces. The spatial reconstruction "
        "analysis confirms that the proposed optimization yields a 26.5% reduction in peak hotspot reconstruction error relative to current "
        "municipal infrastructure.\n\n"
        "Internal and External Validity: Metaheuristic stochasticity was controlled through 30 independent experimental runs with distinct "
        "pseudo-random seeds, supported by non-parametric Friedman and Wilcoxon signed-rank hypothesis testing. Because the snapping and greedy "
        "spatial repair operators rely on general Euclidean distance formulations rather than site-specific heuristics, the proposed framework "
        "is readily transferable to other rapidly expanding metropolitan centers across Southeast Asia, including Hanoi, Bangkok, and Jakarta."
    )

    # Lưu tài liệu
    doc.save(output_path)
    print(f"\n[Success] Updated {output_path} with English text, figures, and tables.")


if __name__ == "__main__":
    main()
