import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collections import defaultdict

st.set_page_config(
    page_title="Perhitungan Spektrofotometri", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gaya CSS
st.markdown("""
<style>
    .main-content { padding: 20px; }
    .sidebar .sidebar-content { background-color: #f8f9fa; }
    .centered { text-align: center; padding: 30px 0; }
    .judul { font-size: 50px; font-weight: 800; color: #4FC3F7; text-shadow: 1px 1px 2px #00000060; margin-bottom: 20px; }
    .subjudul { font-size: 24px; color: #2E7D32; margin-bottom: 30px; }
    .desc { font-size: 18px; color: #424242; max-width: 800px; margin: auto; line-height: 1.6; text-align: justify; }
    .menu-header { font-size: 24px; font-weight: bold; color: #1976D2; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigasi
with st.sidebar:
    st.markdown('<div class="menu-header">📋 Menu Navigasi</div>', unsafe_allow_html=True)
    menu = st.radio("Pilih Halaman:", [
        "🏠 Beranda",
        "📌 Standar Induk", 
        "📊 Deret Standar",
        "📈 Kurva Kalibrasi",
        "🧪 Kadar Sampel",
        "📖 Tentang Kami"
    ], key="navigation_menu")
    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("- Ikuti urutan menu dari atas ke bawah\n- Gunakan titik (.) untuk desimal")

# Data Massa Atom Relatif
massa_atom = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.01,
    "N": 14.007, "O": 16.00, "F": 18.998, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Tc": 98, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41,
    "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "U": 238.03  # dipersingkat
}

# Parsing rumus kimia
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

# Hitung BM
def hitung_bm(formula):
    parsed = parse_formula(formula)
    if not parsed:
        return None
    total = sum(massa_atom[el] * jumlah for el, jumlah in parsed.items())
    return round(total, 4)
