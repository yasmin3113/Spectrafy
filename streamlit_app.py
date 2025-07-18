import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collections import defaultdict
from streamlit_lottie import st_lottie

# Fungsi untuk load Lottie

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Konfigurasi tampilan
st.set_page_config(
    page_title="Perhitungan Spektrofotometri", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
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
    st.markdown('<div class="menu-header">\ud83d\udccb Menu Navigasi</div>', unsafe_allow_html=True)
    menu = st.radio(
        "Pilih Halaman:",
        [
            "\ud83c\udfe0 Beranda",
            "\ud83d\udccc Standar Induk", 
            "\ud83d\udcca Deret Standar",
            "\ud83d\udcc8 Kurva Kalibrasi",
            "\ud83e\uddea Kadar Sampel",
            "\ud83d\udcd6 Tentang Kami"
        ],
        key="navigation_menu"
    )
    st.markdown("---")
    st.markdown("**\ud83d\udca1 Tips:**")
    st.markdown("- Ikuti urutan menu dari atas ke bawah")
    st.markdown("- Pastikan input data sudah benar")
    st.markdown("- Gunakan titik (.) untuk desimal")

# Data massa atom
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
    formula = formula.replace("\u00b7", ".")
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

# ===========================
# MAIN CONTENT STARTS HERE
# ===========================
if menu == "\ud83c\udfe0 Beranda":
    st.markdown("""
    <div class='centered'>
        <div class='judul'>Perhitungan Spektrofotometri</div>
        <div class='subjudul'>\ud83d\udcda Selamat Datang di Aplikasi Kami!</div>
        <div class='desc'>
            Aplikasi ini membantu Anda melakukan perhitungan spektrofotometri secara lengkap dan sistematis. 
            Mulai dari pembuatan larutan standar induk, deret standar, pembuatan kurva kalibrasi, 
            hingga perhitungan kadar sampel berdasarkan absorbansi yang terukur.<br><br>
            <strong>Fitur Utama:</strong><br>
            \u2705 Perhitungan otomatis BM/Mr dari rumus kimia<br>
            \u2705 Pembuatan larutan standar dari zat padat atau larutan pekat<br>
            \u2705 Perhitungan deret standar dengan berbagai konsentrasi<br>
            \u2705 Pembuatan kurva kalibrasi dengan regresi linear<br>
            \u2705 Analisis kadar sampel dengan statistik lengkap<br><br>
            <em>Silakan pilih menu di sidebar untuk memulai perhitungan!</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    lottie_json = load_lottieurl("https://lottie.host/6d628372-b83e-4256-a791-e4c68204bd34/ee70UtmL8I.json")
    if lottie_json:
        st_lottie(lottie_json, height=200, key="navigasi")

# (Seluruh menu lain seperti Standar Induk, Deret Standar, Kurva Kalibrasi, Kadar Sampel, dan Tentang Kami
# tetap sama seperti yang sudah Anda tulis sebelumnya. Jika ingin saya lengkapi semua halaman dalam satu file,
# beri tahu saya dan saya akan lanjutkan penulisan ke bawah.)
