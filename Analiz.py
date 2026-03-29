import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. VERİ SETİ VE KONFİGÜRASYONLAR ---
# FM24 stili benchmarks ve column_map (Değişmedi)
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

# --- YARDIMCI FONKSİYONLAR ---
def clean_val(val):
    try:
        if pd.isna(val): return 0.0
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

# --- TABLO RENKLENDİRME FONKSİYONLARI ---
def get_color_style(val, metric, pos_group):
    v = clean_val(val)
    thresh = benchmarks[pos_group].get(metric)
    if not thresh: return ""
    
    # Eşik değerlerin azalan mı yoksa artan mı olduğunu kontrol et
    # Örn: Possession Lost [12.01, 8.18, 4.90, 4.05] (is_reverse = True)
    is_reverse = thresh[0] > thresh[-1] 
    
    if is_reverse:
        if v <= thresh[3]: return "background-color: #00008B; color: white; font-weight: bold;" # Elit
        if v <= thresh[2]: return "background-color: #006400; color: white; font-weight: bold;" # İyi
        if v <= thresh[1]: return "background-color: #B8860B; color: white; font-weight: bold;" # Ortalama
        return "background-color: #8B0000; color: white; font-weight: bold;" # Zayıf
    else:
        if v >= thresh[3]: return "background-color: #00008B; color: white; font-weight: bold;" # Elit
        if v >= thresh[2]: return "background-color: #006400; color: white; font-weight: bold;" # İyi
        if v >= thresh[1]: return "background-color: #B8860B; color: white; font-weight: bold;" # Ortalama
        return "background-color: #8B0000; color: white; font-weight: bold;" # Zayıf

def style_stats_dataframe(df_to_style, pos_group):
    styles = pd.DataFrame('', index=df_to_style.index, columns=df_to_style.columns)
    for metric in df_to_style.index:
        for player in df_to_style.columns:
            styles.at[metric, player] = get_color_style(df_to_style.at[metric, player], metric, pos_group)
    return styles

# --- 2. PLOTLY İNTERAKTİF RADAR TASARIMI (MODERN VE FM24 SİTİLİ) ---
def draw_radar_pro(players_data, metrics, pos_group):
    fig = go.Figure()
    
    # Plotly'de poligonun kapanması için ilk öğeyi sona eklememiz gerekir.
    theta_labels = [f"{m}<br>({benchmarks[pos_group][m][3]})" for m in metrics]
    theta_ext = theta_labels + [theta_labels[0]]
    
    # FM24 Stili Arka Plan Renk Halkaları (Neon Transparan)
    bg_levels = [
        (100, "rgba(0, 0, 139, 0.25)", "Elit Bölge"),     # Mavi
        (75, "rgba(0, 100, 0, 0.3)", "İyi Bölge"),        # Yeşil
        (50, "rgba(184, 134, 11, 0.35)", "Ortalama"),     # Hardal
        (25, "rgba(139, 0, 0, 0.4)", "Zayıf Bölge")       # Kırmızı
    ]
    
    for val, color, name in bg_levels:
        fig.add_trace(go.Scatterpolar(
            r=[val] * len(theta_ext),
            theta=theta_ext,
            fill='toself',
            fillcolor=color,
            line=dict(color='rgba(255,255,255,0)'), # Çizgisiz
            name=name,
            hoverinfo='skip',
            showlegend=False
        ))

    # Geniş renk havuzu (30 oyuncuya kadar belirgin kalır)
    line_colors = [
        "#00FFFF", "#FF00FF", "#ADFF2F", "#FFA500", "#FFD700", 
        "#00FA9A", "#1E90FF", "#FF69B4", "#CD5C5C", "#8A2BE2",
        "#00BFFF", "#32CD32", "#FF4500", "#DA70D6", "#F0E68C"
    ]
    
    for i, (name, values) in enumerate(players_data.items()):
        r_vals = values.tolist() + [values.tolist()[0]]
        color = line_colors[i % len(line_colors)]
        
        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_ext,
            fill='none', 
            line=dict(color=color, width=3),
            marker=dict(color=color, size=7, line=dict(color='white', width=1)),
            name=name,
            # FM stili interaktif hover bilgisi
            hovertemplate="<b>%{theta}</b><br>Normalleştirilmiş Skor: %{r:.1f}<extra></extra>"
        ))

    # FM Stili Koyu Tema ve Neon Aksanlar
    fig.update_layout(
        polar=dict(
            bgcolor='#0E1117',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False, # Sayısal değerleri gizledik ki sade dursun
                gridcolor='rgba(255, 255, 255, 0.2)',
                gridwidth=1
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)',
                linecolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(size=12, color="#E0E0E0", family="Arial")
            )
        ),
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(color="white", size=11),
            bgcolor="rgba(0,0,0,0)"
        )
    )
    return fig

# --- 3. STREAMLIT UI (DEĞİŞMEDİ) ---
st.set_page_config(layout="wide", page_title="FM26 Scout Pro")

# FM stili Kutu CSS'i
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
    
    # KONTROL PANELİ
    st.sidebar.subheader("🛠️ Panel")
    pos_group = st.sidebar.selectbox("Pozisyon Seçimi", ["DEF", "MID", "FWD", "GK"])
    selected_players = st.sidebar.multiselect("Oyuncuları Kıyasla", df["Player"].unique(), max_selections=30)
    metrics = [m for m in benchmarks[pos_group].keys() if column_map[m] in df.columns]

    st.sidebar.write("---")
    st.sidebar.subheader("🎨 Renk Kılavuzu")
    st.sidebar.markdown('<div class="level-box" style="background-color: #00008B; color: white;">ELİT</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="level-box" style="background-color: #006400; color: white;">İYİ</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="level-box" style="background-color: #B8860B; color: white;">ORTALAMA</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="level-box" style="background-color: #8B0000; color: white;">ZAYIF</div>', unsafe_allow_html=True)

    if selected_players:
        num_players = len(selected_players)
        
        # Sınırlandırılmış dinamik sütun oranları (Negatif değer veya layout bozulmasını engeller)
        plot_weight = max(1.0, 3.0 - (num_players * 0.05)) 
        table_weight = max(2.0, 1.0 + (num_players * 0.3)) 
        
        col_plot, col_val = st.columns([plot_weight, table_weight])
        
        with col_plot:
            plot_data = {}
            for p in selected_players:
                row = df[df["Player"] == p].iloc[0]
                norm_vals = [get_norm(row.get(column_map[m], 0), benchmarks[pos_group][m], m in ["Goals Conceded", "Possession Lost"]) for m in metrics]
                plot_data[p] = pd.Series(norm_vals)
            
            # Matplotlib yerine Plotly Chart'ı render ediyoruz
            fig = draw_radar_pro(plot_data, metrics, pos_group)
            st.plotly_chart(fig, use_container_width=True)

        with col_val:
            st.subheader("📊 Oyuncu İstatistikleri")
            stats_data = []
            for m in metrics:
                row_dict = {"Metrik": m}
                for p in selected_players:
                    val = clean_val(df[df["Player"] == p].iloc[0].get(column_map[m], 0))
                    row_dict[p] = val
                stats_data.append(row_dict)
            
            # DataFrame'i oluştur, metriği index yap ve stili uygula
            stat_df = pd.DataFrame(stats_data).set_index("Metrik")
            styled_stat_df = stat_df.style.apply(lambda x: style_stats_dataframe(stat_df, pos_group), axis=None).format("{:.2f}")
            
            # (Satır sayısı + 1 başlık) * satır yüksekliği (~36px) = Kusursuz dikey yükseklik
            dynamic_height = int((len(metrics) + 1) * 36)
            
            # Gelişmiş tabloyu render et
            st.dataframe(styled_stat_df, use_container_width=True, height=dynamic_height)

    else:
        st.info("Lütfen sol panelden en az 1 oyuncu seçin.")
