import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collections import defaultdict

# Konfigurasi tampilan
st.set_page_config(
    page_title="Perhitungan Spektrofotometri", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling yang lebih baik
st.markdown("""
<style>
    .main-content {
        padding: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stRadio > div {
        padding: 10px 0;
    }
    .centered {
        text-align: center;
        padding: 30px 0;
    }
    .judul {
        font-size: 50px;
        font-weight: 800;
        color: #4FC3F7;
        text-shadow: 1px 1px 2px #00000060;
        margin-bottom: 20px;
    }
    .subjudul {
        font-size: 24px;
        color: #2E7D32;
        margin-bottom: 30px;
    }
    .desc {
        font-size: 18px;
        color: #424242;
        max-width: 800px;
        margin: auto;
        line-height: 1.6;
        text-align: justify;
    }
    .img-container {
        margin: 20px 0 30px 0;
    }
    .menu-header {
        font-size: 24px;
        font-weight: bold;
        color: #1976D2;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="menu-header">📋 Menu Navigasi</div>', unsafe_allow_html=True)
    
    menu = st.radio(
        "Pilih Halaman:",
        [
            "🏠 Beranda",
            "📌 Standar Induk", 
            "📊 Deret Standar",
            "📈 Kurva Kalibrasi",
            "🧪 Kadar Sampel",
            "📖 Tentang Kami"
        ],
        key="navigation_menu"
    )
    
    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("- Ikuti urutan menu dari atas ke bawah")
    st.markdown("- Pastikan input data sudah benar")
    st.markdown("- Gunakan titik (.) untuk desimal")

# Data Massa Atom Relatif
massa_atom = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.01,
    "N": 14.007, "O": 16.00, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Tc": 98, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41,
    "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50,
    "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49,
    "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08,
    "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209,
    "At": 210, "Rn": 222, "Fr": 223, "Ra": 226, "Ac": 227, "Th": 232.04, "Pa": 231.04,
    "U": 238.03, "Np": 237, "Pu": 244, "Am": 243, "Cm": 247, "Bk": 247, "Cf": 251,
    "Es": 252, "Fm": 257, "Md": 258, "No": 259, "Lr": 266, "Rf": 267, "Db": 268,
    "Sg": 269, "Bh": 270, "Hs": 277, "Mt": 278, "Ds": 281, "Rg": 282, "Cn": 285,
    "Fl": 289, "Lv": 293, "Ts": 294, "Og": 294
}

def parse_formula(formula):
    formula = formula.replace("·", ".")
    parts = formula.split(".")
    total_elements = defaultdict(int)
    
    def parse(part, multiplier=1):
        stack = []
        i = 0
        while i < len(part):
            if part[i] == "(":
                stack.append(({}, multiplier))
                i += 1
            elif part[i] == ")":
                i += 1
                num = ""
                while i < len(part) and part[i].isdigit():
                    num += part[i]
                    i += 1
                group_multiplier = int(num) if num else 1
                group_dict, _ = stack.pop()
                for el, count in group_dict.items():
                    if stack:
                        stack[-1][0][el] = stack[-1][0].get(el, 0) + count * group_multiplier
                    else:
                        total_elements[el] += count * group_multiplier * multiplier
            else:
                match = re.match(r'([A-Z][a-z]?)(\d*)', part[i:])
                if not match:
                    return None
                el = match.group(1)
                count = int(match.group(2)) if match.group(2) else 1
                i += len(match.group(0))
                if el not in massa_atom:
                    return None
                if stack:
                    stack[-1][0][el] = stack[-1][0].get(el, 0) + count
                else:
                    total_elements[el] += count * multiplier
    
    for part in parts:
        match = re.match(r'^(\d+)([A-Z(].*)', part)
        mul = int(match.group(1)) if match else 1
        formula_part = match.group(2) if match else part
        parse(formula_part, multiplier=mul)
    
    return total_elements

def hitung_bm(formula):
    parsed = parse_formula(formula)
    if not parsed:
        return None
    total = sum(massa_atom[el] * jumlah for el, jumlah in parsed.items())
    return round(total, 4)

# Main Content Area
with st.container():
    if menu == "🏠 Beranda":
        st.markdown("""
        <div class='centered'>
            <div class='judul'>Perhitungan Spektrofotometri</div>
            <div class='subjudul'>📚 Selamat Datang di Aplikasi Kami!</div>
                <img src='https://cdn-icons-png.flaticon.com/512/2933/2933116.png' width='150'/>
            </div>
            <div class='desc'>
                Aplikasi ini membantu Anda melakukan perhitungan spektrofotometri secara lengkap dan sistematis. 
                Mulai dari pembuatan larutan standar induk, deret standar, pembuatan kurva kalibrasi, 
                hingga perhitungan kadar sampel berdasarkan absorbansi yang terukur.
                <br><br>
                <strong>Fitur Utama:</strong><br>
                ✅ Perhitungan otomatis BM/Mr dari rumus kimia<br>
                ✅ Pembuatan larutan standar dari zat padat atau larutan pekat<br>
                ✅ Perhitungan deret standar dengan berbagai konsentrasi<br>
                ✅ Pembuatan kurva kalibrasi dengan regresi linear<br>
                ✅ Analisis kadar sampel dengan statistik lengkap<br>
                <br>
                <em>Silakan pilih menu di sidebar untuk memulai perhitungan!</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif menu == "📌 Standar Induk":
        st.header("📌 1. Pembuatan Larutan Standar Induk")
        
        # Input rumus kimia otomatis hitung BM
        col1, col2 = st.columns(2)
        with col1:
            rumus_garam = st.text_input("🔹 Rumus Kimia Garam", placeholder="Contoh: NaCl")
            bm_garam = hitung_bm(rumus_garam) if rumus_garam else 0.0
            st.number_input("BM Garam (g/mol)", value=bm_garam, format="%.4f", key="bm_garam")
        
        with col2:
            rumus_senyawa = st.text_input("🔹 Rumus Kimia Senyawa", placeholder="Contoh: Cl")
            bm_senyawa = hitung_bm(rumus_senyawa) if rumus_senyawa else 0.0
            st.number_input("BM Senyawa (g/mol)", value=bm_senyawa, format="%.4f", key="bm_senyawa")
        
        # Perhitungan metode
        metode = st.radio("Metode Pembuatan Larutan:", ["Dari zat padat", "Dari larutan pekat"])
        
        if metode == "Dari zat padat":
            ppm_str = st.text_input("Konsentrasi (mg/L)", placeholder="100")
            V_str = st.text_input("Volume (mL)", placeholder="100")
            if st.button("Hitung Massa Zat Padat"):
                try:
                    ppm = float(ppm_str)
                    V = float(V_str)
                    massa = ((bm_garam * ppm * (V / 1000)) / bm_senyawa) / 1000
                    st.success(f"🔹 Massa zat padat yang dibutuhkan: {massa:.4f} gram")
                except:
                    st.error("Mohon isi semua nilai dengan benar.")
        
        else:
            C1_str = st.text_input("Konsentrasi Pekat (mg/L)", placeholder="1000")
            C2_str = st.text_input("Konsentrasi Target (mg/L)", placeholder="100")
            V2_str = st.text_input("Volume Target (mL)", placeholder="100")
            if st.button("Hitung Volume Pekat yang Dibutuhkan"):
                try:
                    C1 = float(C1_str)
                    C2 = float(C2_str)
                    V2 = float(V2_str)
                    V1 = (C2 * V2) / C1
                    st.success(f"🔹 Ambil {V1:.2f} mL larutan pekat, encerkan hingga {V2} mL")
                except:
                    st.error("Input tidak valid.")
    
    elif menu == "📊 Deret Standar":
        st.header("📊 2. Deret Standar dari Larutan Induk")
        
        vol_total_str = st.text_input("Volume labu masing-masing larutan (mL)", placeholder="Contoh: 10")
        kons_induk_str = st.text_input("Konsentrasi larutan induk (mol/L)", placeholder="Contoh: 1.0")
        konsen_str = st.text_input("Deret konsentrasi yang diinginkan (pisahkan dengan koma)", placeholder="Contoh: 0.2, 0.4, 0.6")
        
        if st.button("Hitung Volume Deret Standar"):
            try:
                vol_total = float(vol_total_str)
                kons_induk = float(kons_induk_str)
                kons_list = [float(i.strip()) for i in konsen_str.split(",") if i.strip() != ""]
                data = []
                for C2 in kons_list:
                    V1 = (C2 * vol_total) / kons_induk
                    data.append([C2, V1, vol_total - V1])
                df = pd.DataFrame(data, columns=["Konsentrasi (mg/L)", "Volume Induk (mL)", "Volume Pelarut (mL)"])
                st.dataframe(df, use_container_width=True)
            except:
                st.error("Periksa kembali format input Anda. Pastikan semua angka valid dan tidak kosong.")
    
    elif menu == "📈 Kurva Kalibrasi":
        st.header("📈 3. Kurva Kalibrasi & Regresi")
        
        kons_cal = st.text_input("Konsentrasi Standar (mg/L)", placeholder="Contoh: 0.2, 0.4, 0.6")
        abs_cal = st.text_input("Absorbansi Standar", placeholder="Contoh: 0.25, 0.48, 0.75")
        
        if st.button("Buat Kurva Kalibrasi"):
            try:
                x = np.array([float(i.strip()) for i in kons_cal.split(",") if i.strip() != ""])
                y = np.array([float(i.strip()) for i in abs_cal.split(",") if i.strip() != ""])
                
                if len(x) != len(y):
                    st.error("Jumlah konsentrasi dan absorbansi tidak sama.")
                else:
                    model = LinearRegression()
                    model.fit(x.reshape(-1, 1), y)
                    y_pred = model.predict(x.reshape(-1, 1))
                    r2 = r2_score(y, y_pred)
                    
                    b = model.coef_[0]
                    a = model.intercept_
                    st.session_state["regresi_a"] = a
                    st.session_state["regresi_b"] = b
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.regplot(x=x, y=y, ax=ax, ci=None, line_kws={"color": "red"})
                    ax.set_title("Kurva Kalibrasi UV-Vis", fontsize=16)
                    ax.set_xlabel("Konsentrasi (mg/L)", fontsize=12)
                    ax.set_ylabel("Absorbansi", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"Persamaan regresi: y = {b:.4f}x + {a:.4f}")
                    with col2:
                        st.info(f"Koefisien Determinasi R² = {r2:.4f}")
            
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    
    elif menu == "🧪 Kadar Sampel":
        st.header("🧪 4. Hitung Kadar dari Absorbansi")
        
        absorb_str = st.text_area("Masukkan absorbansi sampel (pisahkan dengan koma)", 
                                 placeholder="Contoh: 0.234, 0.245, 0.238")
        
        # Ambil nilai a dan b dari session_state kalau ada
        default_a = st.session_state.get('regresi_a', 0.0123)
        default_b = st.session_state.get('regresi_b', 1.234)
        default_regresi = f"y = {default_a:.4f} + {default_b:.4f}x"
        regresi = st.text_input("Persamaan regresi kalibrasi (format: y = a + bx)", default_regresi)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            faktor_pengencer_str = st.text_input("Faktor Pengenceran", placeholder="Contoh: 10")
        with col2:
            volume_labu_str = st.text_input("Volume Labu Takar (mL)", placeholder="Contoh: 100")
        with col3:
            bobot_sample_str = st.text_input("Bobot Sampel (gram)", placeholder="Contoh: 1.0000")
        
        if st.button("Hitung Kadar Sampel"):
            try:
                absorb = np.array([float(i.strip()) for i in absorb_str.split(",") if i.strip()])
                
                # Validasi input numerik
                if not faktor_pengencer_str.strip() or not volume_labu_str.strip() or not bobot_sample_str.strip():
                    st.warning("Mohon lengkapi semua input angka terlebih dahulu.")
                    st.stop()
                
                faktor_pengencer = float(faktor_pengencer_str)
                volume_labu = float(volume_labu_str)
                bobot_sample = float(bobot_sample_str)
                
                if faktor_pengencer <= 0 or volume_labu <= 0 or bobot_sample <= 0:
                    st.error("Semua nilai harus lebih dari 0.")
                    st.stop()
                
                # Parsing regresi dengan regex
                match = re.search(r"y\s*=\s*([-+]?\d*\.?\d+)\s*\+\s*([-+]?\d*\.?\d+)x", regresi)
                if match:
                    a = float(match.group(1))
                    b = float(match.group(2))
                else:
                    st.error("Format persamaan regresi salah. Gunakan format: y = a + bx")
                    st.stop()
                
                # Hitung konsentrasi dan kadar
                konsentrasi_terukur = (absorb - a) / b
                kadar_sampel = (konsentrasi_terukur * faktor_pengencer * volume_labu / 1000) / bobot_sample * 1000
                
                # Statistik
                rata2 = np.mean(kadar_sampel)
                std = np.std(kadar_sampel, ddof=1)
                rsd = (std / rata2) * 100
                rpd = (np.max(kadar_sampel) - np.min(kadar_sampel)) / rata2 * 100
                
                # Tabel detail
                df_sampel = pd.DataFrame({
                    "Absorbansi": absorb,
                    "Konsentrasi Terukur (mg/L)": np.round(konsentrasi_terukur, 4),
                    "Kadar Sampel (mg/kg)": np.round(kadar_sampel, 4)
                })
                
                # Tabel ringkasan
                df_ringkasan = pd.DataFrame({
                    "Parameter": [
                        "Rata-rata Kadar (mg/kg)",
                        "Simpangan Baku (mg/kg)",
                        "RSD (%)",
                        "RPD (%)"
                    ],
                    "Nilai": [
                        f"{rata2:.4f}",
                        f"{std:.4f}",
                        f"{rsd:.2f}",
                        f"{rpd:.2f}"
                    ]
                })
                
                # Output
                st.subheader("📊 Data Sampel")
                st.dataframe(df_sampel, use_container_width=True)
                
                st.subheader("📋 Ringkasan Perhitungan")
                st.dataframe(df_ringkasan, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif menu == "📖 Tentang Kami":
    st.header("📖 Tentang Kami")
    
    # Deskripsi proyek
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #1f77b4; margin-top: 0;">Tentang Proyek Ini</h3>
        <p>Aplikasi ini dikembangkan sebagai bagian dari tugas kelompok yang bertujuan untuk 
        [jelaskan tujuan aplikasi Anda di sini]. Proyek ini merupakan hasil kolaborasi tim 
        yang terdiri dari 5 mahasiswa yang berdedikasi.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tim pengembang
    st.subheader("👥 Tim Pengembang")
    
    anggota = [
        {"nama": "Devi Triana Rahmadina", "nim": "2460352"},
        {"nama": "Indra Alfin Nur Riski", "nim": "2460389"},
        {"nama": "Muhammad Diptarrama Rids", "nim": "2460436"},
        {"nama": "Saskia Arizqa Syaakirah", "nim": "2460511"},
        {"nama": "Yasmin Anbarcitra", "nim": "2460536"},
    ]
    
    # Tampilkan dalam format list yang clean
    for i, member in enumerate(anggota, 1):
        col1, col2, col3 = st.columns([1, 4, 2])
        
        with col1:
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: center; 
                        width: 40px; height: 40px; border-radius: 50%; 
                        background: linear-gradient(45deg, #1f77b4, #ff7f0e); 
                        color: white; font-weight: bold; font-size: 18px;">
                {i}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 10px 0;">
                <h4 style="margin: 0; color: #2c3e50; font-size: 18px;">{member['nama']}</h4>
                <p style="margin: 0; color: #7f8c8d; font-size: 14px;">NIM: {member['nim']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="padding: 10px 0; text-align: right;">
                <span style="background: #e8f4f8; color: #1f77b4; padding: 4px 8px; 
                           border-radius: 12px; font-size: 12px; font-weight: 500;">
                    Anggota
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        if i < len(anggota):
            st.markdown('<hr style="margin: 15px 0; border: 0.5px solid #ecf0f1;">', unsafe_allow_html=True)
    
    # Informasi tambahan
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠️ Teknologi")
        st.markdown("""
        - **Framework:** Streamlit
        - **Bahasa:** Python
        - **Library:** Pandas, NumPy, Matplotlib
        - **Database:** [sesuaikan dengan project]
        """)
    
    with col2:
        st.subheader("📅 Timeline")
        st.markdown("""
        - **Mulai:** [tanggal mulai]
        - **Selesai:** [tanggal selesai]
        - **Durasi:** [durasi pengerjaan]
        - **Status:** ✅ Selesai
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-style: italic;">
        <p>Terima kasih telah menggunakan aplikasi kami! 🙏</p>
        
        st.markdown("### 👥 Tim Pengembang")
        
        # Menggunakan grid layout untuk menampilkan anggota
        cols = st.columns(3)
        for idx, anggota_data in enumerate(anggota):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 15px;
                    margin: 10px 0;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h4 style="margin: 0 0 10px 0; font-size: 16px;">{anggota_data['nama']}</h4>
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;"><strong>NIM:</strong> {anggota_data['nim']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        ### 🎯 Tujuan Aplikasi
        Aplikasi ini dikembangkan untuk membantu mahasiswa dan praktisi kimia dalam melakukan perhitungan 
        spektrofotometri dengan mudah dan akurat. Semua perhitungan dilakukan secara otomatis dengan 
        interface yang user-friendly.
