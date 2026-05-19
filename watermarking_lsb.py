import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
import os


# ==============================================================
# FUNGSI 1: Load gambar
# ==============================================================
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    img = Image.open(path).convert('RGB')
    return img


# ==============================================================
# FUNGSI 2: Buat watermark biner
# ==============================================================
def create_binary_watermark(width, height, text="WATERMARK"):
    wm = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(wm)

    font_size = max(20, min(width, height) // 10)
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/Arial.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size=font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=255, font=font)

    wm_array = np.array(wm)
    wm_binary = (wm_array > 128).astype(np.uint8)
    return wm_binary


# ==============================================================
# FUNGSI 3: Sisipkan watermark (LSB)
# ==============================================================
def embed_lsb(image, watermark):
    img_array = np.array(image).copy()
    h, w = watermark.shape
    h = min(h, img_array.shape[0])
    w = min(w, img_array.shape[1])
    img_array[:h, :w, 0] = (img_array[:h, :w, 0] & 0xFE) | watermark[:h, :w]
    return Image.fromarray(img_array)


# ==============================================================
# FUNGSI 4: Ekstrak watermark (LSB)
# ==============================================================
def extract_lsb(image, wm_height, wm_width):
    img_array = np.array(image)
    h = min(wm_height, img_array.shape[0])
    w = min(wm_width, img_array.shape[1])
    extracted = (img_array[:h, :w, 0] & 1).astype(np.uint8)
    return extracted


# ==============================================================
# FUNGSI 5: Hitung BER
# ==============================================================
def calculate_ber(original_wm, extracted_wm):
    h = min(original_wm.shape[0], extracted_wm.shape[0])
    w = min(original_wm.shape[1], extracted_wm.shape[1])
    total_bits = h * w
    error_bits = np.sum(original_wm[:h, :w] != extracted_wm[:h, :w])
    return error_bits / total_bits


# ==============================================================
# FUNGSI 6: Kompres JPEG
# ==============================================================
def jpeg_compress_decompress(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


# ==============================================================
# FUNGSI 7: Hitung PSNR
# ==============================================================
def calculate_psnr(original, watermarked):
    orig = np.array(original).astype(float)
    wm = np.array(watermarked).astype(float)
    mse = np.mean((orig - wm) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


# ==============================================================
# PIPELINE UTAMA
# ==============================================================
def run_watermarking_pipeline(image_path, output_dir='output', watermark_text='WATERMARK'):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 55)
    print("  SISTEM MULTIMEDIA - WATERMARKING LSB")
    print("=" * 55)

    # STEP 1: Load gambar
    print("\n[1] Memuat gambar asli...")
    original = load_image(image_path)
    W, H = original.size
    print(f"    Ukuran gambar: {W} x {H} piksel")

    # STEP 2: Buat watermark biner
    print(f"\n[2] Membuat watermark biner: '{watermark_text}'")
    watermark = create_binary_watermark(W, H, text=watermark_text)
    wm_img = Image.fromarray(watermark * 255)
    wm_img.save(os.path.join(output_dir, '1_watermark_biner.png'))

    # STEP 3: Sisipkan watermark
    print("\n[3] Menyisipkan watermark ke foto (LSB)...")
    watermarked = embed_lsb(original, watermark)
    watermarked.save(os.path.join(output_dir, '2_foto_ber_watermark.png'))
    psnr_embed = calculate_psnr(original, watermarked)
    print(f"    PSNR: {psnr_embed:.2f} dB (>40 dB = tidak terlihat secara visual)")

    # STEP 4: Verifikasi ekstraksi TANPA kompresi
    print(f"\n[4] Verifikasi ekstraksi TANPA kompresi (PNG, lossless):")
    extracted_no_compress = extract_lsb(watermarked, H, W)
    ber_no_compress = calculate_ber(watermark, extracted_no_compress)
    print(f"    BER = {ber_no_compress:.6f} (0.0 = sempurna)")
    Image.fromarray(extracted_no_compress * 255).save(
        os.path.join(output_dir, '3_extracted_tanpa_kompresi.png'))

    # ============================================================
    # VISUALISASI UTAMA: Bukti watermark berhasil disisipkan & diekstrak
    # ============================================================
    print("\n[5] Membuat visualisasi bukti penyisipan & ekstraksi...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Baris atas: proses penyisipan
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('(a) Foto Asli (Cover Image)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(watermark, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'(b) Watermark Biner\n"{watermark_text}"', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(watermarked)
    axes[0, 2].set_title(f'(c) Foto + Watermark (LSB)\nPSNR = {psnr_embed:.2f} dB', fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')

    # Baris bawah: hasil ekstraksi
    axes[1, 0].imshow(extracted_no_compress, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'(d) Watermark Terekstrak\nTANPA Kompresi\nBER = {ber_no_compress:.6f} ✅ SEMPURNA',
                         fontsize=13, fontweight='bold', color='green')
    axes[1, 0].axis('off')

    # Tambahkan tanda panah / penjelasan di tengah bawah
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.6,
                    'LSB Watermarking:\n\n'
                    '• Imperceptible (tidak terlihat)\n'
                    f'  PSNR = {psnr_embed:.1f} dB ✅\n\n'
                    '• Fragile (tidak tahan kompresi)\n'
                    '  BER > 0.3 saat JPEG diterapkan\n\n'
                    '• Kesimpulan:\n'
                    '  Berhasil disisipkan & diekstrak\n'
                    '  dari gambar tanpa kompresi',
                    transform=axes[1, 1].transAxes,
                    ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                              edgecolor='orange', linewidth=2))

    # Diff image (perbedaan antara asli dan ber-watermark, diperkuat 50x)
    diff = np.abs(np.array(original).astype(float) - np.array(watermarked).astype(float))
    diff_amplified = np.clip(diff * 50, 0, 255).astype(np.uint8)
    axes[1, 2].imshow(diff_amplified)
    axes[1, 2].set_title('(e) Perbedaan Asli vs Ber-Watermark\n(diperkuat 50x)\n— tidak terlihat dengan mata biasa —',
                         fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle('Bukti Keberhasilan Watermarking LSB\n(Invisible & Extractable Tanpa Kompresi)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    bukti_path = os.path.join(output_dir, '6_bukti_watermarking_berhasil.png')
    plt.savefig(bukti_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Disimpan: {bukti_path}")

    # STEP 6: Evaluasi JPEG
    qf_values = [100, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]
    ber_results = []
    extracted_wms = []
    THRESHOLD_BER = 0.45
    first_fail_qf = None

    print(f"\n[6] Evaluasi ketahanan watermark terhadap kompresi JPEG:")
    print(f"    {'QF':<6} {'BER':<10} {'PSNR Kompr.':<15} {'Status'}")
    print("    " + "-" * 50)

    for qf in qf_values:
        compressed = jpeg_compress_decompress(watermarked, quality=qf)
        compressed.save(os.path.join(output_dir, f'compressed_qf{qf:03d}.jpg'), quality=qf)
        extracted = extract_lsb(compressed, H, W)
        extracted_wms.append((qf, extracted))
        Image.fromarray(extracted * 255).save(
            os.path.join(output_dir, f'extracted_wm_qf{qf:03d}.png'))
        ber = calculate_ber(watermark, extracted)
        psnr_comp = calculate_psnr(original, compressed)
        ber_results.append(ber)
        if ber >= THRESHOLD_BER and first_fail_qf is None:
            first_fail_qf = qf
        status = "Dapat diekstrak" if ber < THRESHOLD_BER else "Tidak dapat diekstrak"
        mark = "✅" if ber < THRESHOLD_BER else "❌"
        print(f"    QF={qf:<4} BER={ber:.4f}   PSNR={psnr_comp:.1f} dB   {mark} {status}")

    print()
    if first_fail_qf:
        print(f"  >>> Watermark TIDAK DAPAT diekstrak mulai QF = {first_fail_qf}")
    else:
        print(f"  >>> Watermark dapat diekstrak di semua QF yang diuji.")

    # STEP 7: Grid ekstraksi per QF
    print("\n[7] Membuat grid visualisasi ekstraksi per QF...")
    n_cols = 4
    n_rows = -(-len(qf_values) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    for i, (qf, ext_wm) in enumerate(extracted_wms):
        ber = ber_results[i]
        status = "Dapat diekstrak" if ber < THRESHOLD_BER else "Tidak dapat diekstrak"
        mark = "✅" if ber < THRESHOLD_BER else "❌"
        axes[i].imshow(ext_wm, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'QF = {qf}\nBER = {ber:.3f}  {mark}\n{status}', fontsize=10)
        axes[i].axis('off')
    for j in range(len(qf_values), len(axes)):
        axes[j].axis('off')

    plt.suptitle('Watermark Terekstrak pada Berbagai JPEG Quality Factor\n(LSB Watermarking)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    grid_path = os.path.join(output_dir, '3_hasil_ekstraksi_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Disimpan: {grid_path}")

    # STEP 8: Grafik BER vs QF
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['green' if b < THRESHOLD_BER else 'red' for b in ber_results]
    ax.bar(range(len(qf_values)), ber_results, color=colors, alpha=0.7, width=0.6)
    ax.plot(range(len(qf_values)), ber_results, 'ko-', linewidth=1.5, markersize=5)
    ax.axhline(y=THRESHOLD_BER, color='orange', linestyle='--', linewidth=2,
               label=f'Threshold BER = {THRESHOLD_BER}')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5,
               label='BER = 0.5 (pure noise / acak)')
    ax.axhline(y=0.0, color='green', linestyle=':', linewidth=1.5,
               label='BER = 0.0 (ekstraksi sempurna)')

    # Tambahkan titik "tanpa kompresi"
    ax.scatter([-0.5], [ber_no_compress], color='blue', zorder=5, s=100,
               label=f'Tanpa kompresi (PNG): BER = {ber_no_compress:.4f}')

    ax.set_xticks(range(len(qf_values)))
    ax.set_xticklabels([str(qf) for qf in qf_values])
    ax.set_xlabel('JPEG Quality Factor (QF)  ←  lebih rendah = kompresi lebih kuat', fontsize=11)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=11)
    ax.set_title('Evaluasi Ketahanan Watermark LSB terhadap Kompresi JPEG\n'
                 '(Hijau = dapat diekstrak, Merah = tidak dapat diekstrak)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, max(ber_results) + 0.15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ber_path = os.path.join(output_dir, '4_grafik_ber_vs_qf.png')
    plt.savefig(ber_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Disimpan: {ber_path}")

    # STEP 9: Ringkasan akhir
    print("\n" + "=" * 55)
    print("  RINGKASAN HASIL")
    print("=" * 55)
    print(f"  Foto asli              : {image_path} ({W}x{H})")
    print(f"  Watermark              : '{watermark_text}'")
    print(f"  PSNR setelah embed     : {psnr_embed:.2f} dB (invisible ✅)")
    print(f"  BER tanpa kompresi     : {ber_no_compress:.6f} (sempurna ✅)")
    if first_fail_qf:
        print(f"  Watermark gagal mulai  : QF = {first_fail_qf}")
    else:
        print(f"  Watermark kuat di semua QF yang diuji")
    print(f"\n  Output di folder: {output_dir}/")
    print("  File penting untuk laporan:")
    print("    6_bukti_watermarking_berhasil.png  ← BUKTI UTAMA")
    print("    3_hasil_ekstraksi_grid.png          ← grid per QF")
    print("    4_grafik_ber_vs_qf.png              ← grafik evaluasi")
    print("=" * 55)


# ==============================================================
# JALANKAN
# ==============================================================
if __name__ == '__main__':
    IMAGE_PATH = 'foto_sismul.jpg'   # ← ganti nama file fotomu
    WATERMARK_TEXT = '18224086'      # ← ganti dengan NIM/namamu

    run_watermarking_pipeline(
        image_path=IMAGE_PATH,
        output_dir='output',
        watermark_text=WATERMARK_TEXT
    )
