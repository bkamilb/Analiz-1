import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. VERİ SETİ ---
benchmarks = {
    "GK": {
        "Goals Conceded": [1.57, 1.45, 1.31, 1.20], "xG Prevented": [-0.16, -0.12, -0.07, -0.02],
        "Possession Won": [7.25, 7.62, 8.17, 8.53], "Passes Attempted": [21.58, 23.26, 24.38, 26.06],
        "Progressive Passes": [0.37, 0.58, 0.74, 0.98], "Possession Lost": [8.21, 6.04, 3.67, 2.35]
    },
    "DEF": {
        "Blocks": [0.56, 0.62, 0.68, 0.75], "Clearances": [0.90, 1.03, 1.13, 1.28], "Interceptions": [2.38, 2.65, 2.88, 3.12],
        "Tackles Attempted": [0.87, 1.07, 2.25, 3.48], "Tackle Win Rate": [72.4, 74.7, 76.7, 79.0],
        "Headers Attempted": [3.15, 3.82, 4.57, 5.35], "Header Win Rate": [52.4, 60.8, 65.6, 70.7],
        "Possession Won": [8.33, 9.09, 9.77, 10.54], "Passes Attempted": [50.96, 56.45, 62.33, 69.88],
        "Progressive Passes": [3.19, 4.44, 6.18, 7.95], "Possession Lost": [12.01, 8.18, 4.90, 4.05],
        "Key Passes": [0.20, 0.29, 0.77, 1.39], "Expected Assists": [0.02, 0.03, 0.06, 0.13],
        "Assists": [0.01, 0.02, 0.06, 0.13], "Shots": [0.25, 0.34, 0.48, 0.68],
        "Expected Goals": [0.01, 0.02, 0.03, 0.04], "Goals": [0.01, 0.02, 0.03, 0.05], "Dribbles": [0.04, 0.08, 0.55, 1.75]
    },
    "MID": {
        "Blocks": [0.27, 0.40, 0.51, 0.61], "Clearances": [0.49, 0.70, 0.84, 1.01], "Interceptions": [1.96, 2.32, 2.52, 2.86],
        "Tackles Attempted": [1.49, 1.98, 2.34, 2.72], "Tackle Win Rate": [67.3, 69.8, 72.4, 74.7],
        "Headers Attempted": [1.67, 2.16, 2.64, 3.45], "Header Win Rate": [27.0, 36.1, 44.8, 54.4],
        "Possession Won": [5.63, 6.92, 7.82, 8.53], "Passes Attempted": [48.62, 54.66, 60.74, 68.83],
        "Progressive Passes": [3.69, 5.02, 5.82, 6.91], "Possession Lost": [10.21, 8.44, 7.15, 6.04],
        "Key Passes": [0.87, 1.20, 1.60, 2.08], "Expected Assists": [0.06, 0.09, 0.13, 0.19],
        "Assists": [0.05, 0.08, 0.13, 0.20], "Shots": [0.88, 1.17, 1.57, 2.02],
        "Expected Goals": [0.04, 0.06, 0.11, 0.22], "Goals": [0.04, 0.07, 0.11, 0.23], "Dribbles": [0.22, 0.37, 0.72, 1.56]
    },
    "FWD": {
        "Blocks": [0.10, 0.16, 0.24, 0.32], "Clearances": [0.17, 0.31, 0.49, 0.64], "Interceptions": [0.90, 1.37, 1.97, 2.34],
        "Tackles Attempted": [0.48, 1.24, 2.34, 2.94], "Tackle Win Rate": [65.3, 70.7, 74.0, 76.6],
        "Headers Attempted": [2.92, 3.60, 4.94, 6.86], "Header Win Rate": [19.5, 26.9, 33.5, 40.3],
        "Possession Won": [2.47, 4.16, 6.52, 7.52], "Passes Attempted": [28.63, 35.10, 43.27, 50.29],
        "Progressive Passes": [0.98, 2.10, 3.55, 4.75], "Possession Lost": [14.95, 12.36, 8.53, 6.12],
        "Key Passes": [1.18, 1.53, 1.95, 2.31], "Expected Assists": [0.12, 0.16, 0.21, 0.26],
        "Assists": [0.12, 0.17, 0.22, 0.27], "Shots": [1.96, 2.22, 2.47, 2.81],
        "Expected Goals": [0.21, 0.26, 0.32, 0.40], "Goals": [0.22, 0.29, 0.37, 0.47], "Dribbles": [0.84, 1.63, 3.43, 5.62]
    }
}

column_map = {
    "Goals Conceded": "Goals Conceded/90", "xG Prevented": "xG Prevented/90",
    "Blocks": "Blk/90", "Clearances": "Clr/90", "Interceptions": "Int/90",
    "Tackles Attempted": "Tck A", "Tackle Win Rate": "Tck R",
    "Headers Attempted": "Aer A/90", "Header Win Rate": "Hdr %",
    "Possession Won": "Poss Won/90", "Passes Attempted": "Ps A/90",
    "Progressive Passes": "Pr passes/90", "Possession Lost": "Poss Lost/90",
    "Dribbles": "Drb/90", "Key Passes": "KP/90", "Expected Assists": "xA/90",
    "Assists": "Asts/90", "Shots": "Shot/90", "Expected Goals": "xG/90", 
    "Goals": "Goals per 90 minutes"
}

def clean_val(val):
    try:
        if isinstance(val, str): val = val.replace('%','').replace(',','.').strip()
        return float(val) if val != "-" and val != "" else 0.0
    except: return 0.0

def get_norm(val, thresh, rev=False):
    v = clean_val(val)
    l, m, g, e = thresh
    if rev:
        if v >= l: return 10
        if v <= e: return 100
        return 10 + (l - v) / (l - e) * 90
    else:
        if v <= l: return (v / l) * 25 if l > 0 else 5
        if v >= e: return 100
        return 25 + (v - l) / (e - l) * 75

# --- 2. RADAR TASARIMI (TEMİZ) ---
def draw_radar_pro(players_data, metrics, pos_group):
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    # RENK HALKALARI
    levels = [(25, "#8B0000"), (50, "#B8860B"), (75, "#006400"), (100, "#00008B")]
    for top, color in reversed(levels):
        ax.fill(angles, [top]*(N+1), color=color, alpha=0.5, zorder=0)

    # OYUNCULAR
    line_colors = ["#00FFFF", "#FF00FF", "#ADFF2F"]
    for i, (name, values) in enumerate(players_data.items()):
        vals = values.tolist()
        vals += vals[:1]
        color = line_colors[i % len(line_colors)]
        ax.plot(angles, vals, color=color, linewidth=4, label=name, marker='o', markersize=7, markeredgecolor='white', zorder=5)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    labels = [f"{m}\n({benchmarks[pos_group][m][3]})" for m in metrics]
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold', color="#E0E0E0")
    
    ax.set_ylim(0, 100)
    ax.set_yticklabels([]) 
    ax.grid(color="#444444", linestyle="--", alpha=0.5)

    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10, facecolor='#1E1E1E', edgecolor='white')
    return fig

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="FM26 Scout Pro")

st.markdown("""
    <style>
    .level-box { padding: 8px; border-radius: 4px; margin-bottom: 4px; font-weight: bold; text-align: center; font-size: 0.9em; }
    .main { background-color: #0E1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 FM26 Profesyonel Scout Analiz Raporu")

file = st.file_uploader("FM Veri Dosyasını Yükle (CSV)", type="csv")

if file:
    df = pd.read_csv(file, sep=";")
    
    col_set, col_plot, col_val = st.columns([1, 2.2, 1.5])
    
    with col_set:
        st.subheader("🛠️ Panel")
        pos_group = st.selectbox("Pozisyon Seçimi", ["DEF", "MID", "FWD", "GK"])
        selected_players = st.multiselect("Oyuncuları Kıyasla", df["Player"].unique(), max_selections=3)
        metrics = [m for m in benchmarks[pos_group].keys() if column_map[m] in df.columns]

        st.write("---")
        st.subheader("🎨 Renk Kılavuzu")
        st.markdown('<div class="level-box" style="background-color: #00008B; color: white;">ELİT</div>', unsafe_allow_html=True)
        st.markdown('<div class="level-box" style="background-color: #006400; color: white;">İYİ</div>', unsafe_allow_html=True)
        st.markdown('<div class="level-box" style="background-color: #B8860B; color: white;">ORTALAMA</div>', unsafe_allow_html=True)
        st.markdown('<div class="level-box" style="background-color: #8B0000; color: white;">ZAYIF</div>', unsafe_allow_html=True)

    with col_plot:
        if selected_players:
            plot_data = {}
            for p in selected_players:
                row = df[df["Player"] == p].iloc[0]
                norm_vals = [get_norm(row[column_map[m]], benchmarks[pos_group][m], m in ["Goals Conceded", "Possession Lost"]) for m in metrics]
                plot_data[p] = pd.Series(norm_vals)
            
            fig = draw_radar_pro(plot_data, metrics, pos_group)
            st.pyplot(fig)
        else:
            st.info("Lütfen sol panelden oyuncu seçin.")

    with col_val:
        if selected_players:
            # TABLO 1: OYUNCU İSTATİSTİKLERİ
            st.subheader("📊 Oyuncu İstatistikleri")
            stats_data = []
            for m in metrics:
                row_dict = {"Metrik": m}
                for p in selected_players:
                    val = clean_val(df[df["Player"] == p].iloc[0][column_map[m]])
                    row_dict[p] = val
                stats_data.append(row_dict)
            st.dataframe(pd.DataFrame(stats_data).set_index("Metrik"), use_container_width=True)

            # TABLO 2: EŞİK DEĞERLERİ (SENİN İSTEDİĞİN FORMAT)
            st.subheader("🎯 Eşik Değerleri (Referans)")
            ref_data = []
            for m in metrics:
                thresh = benchmarks[pos_group][m]
                ref_data.append({
                    "Metrik": m,
                    "Zayıf": thresh[0],
                    "Ortalama": thresh[1],
                    "İyi": thresh[2],
                    "Elit": thresh[3]
                })
            st.dataframe(pd.DataFrame(ref_data).set_index("Metrik"), use_container_width=True)
