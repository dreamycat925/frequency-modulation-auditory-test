import io
import wave
from textwrap import dedent

import numpy as np
import streamlit as st


st.set_page_config(
    page_title="FM聴覚変調検査（Frequency Modulation Auditory Test）",
    page_icon="🎧",
    layout="centered",
)

st.title("🎧 FM聴覚変調検査（Frequency Modulation Auditory Test）")

st.markdown(
    """
**目的**  
周波数変調（Frequency Modulation; FM）の「揺れ」をどの程度知覚できるかを評価するための簡易ツールです。  
Grube et al. (2016) の **2 Hz / 40 Hz FM 検出課題**を参考にした **FM検出タスク**です。

**注意事項**
- 必ず **有線ヘッドホン** を使用してください（Bluetooth / スピーカーは不可）。
- 音量は事前に別の音源で快適レベル（MCL）に調整してから検査してください。
- iPhone / iPad を使用する場合も、Safari ブラウザ＋有線ヘッドホンでの使用を推奨します。
"""
)

st.sidebar.header("⚙️ パラメータ設定")

sr = st.sidebar.number_input("サンプリング周波数 (Hz)", 8000, 48000, 44100, 1000)
freq = st.sidebar.number_input("キャリア周波数 (Hz)", 200, 4000, 500, 100)
dur_ms = st.sidebar.number_input("音の長さ (ms)", 100, 4000, 1000, 100)

# 出力チャネル（両耳 / 左耳のみ / 右耳のみ）
ear = st.sidebar.radio(
    "出力チャネル",
    ["両耳", "左耳のみ", "右耳のみ"],
    index=0,
    help="CFTと同様に、FM刺激を両耳・左耳のみ・右耳のみのいずれかに出力します。",
)

st.sidebar.markdown("### FM周波数（推奨設定＋任意変更）")

# 初期値を 2 Hz にしておく
if "fm_hz" not in st.session_state:
    st.session_state["fm_hz"] = 2.0

# 推奨ボタン（2 Hz / 40 Hz）
bcol1, bcol2 = st.sidebar.columns(2)
with bcol1:
    if st.button("2 Hzに設定"):
        st.session_state["fm_hz"] = 2.0
with bcol2:
    if st.button("40 Hzに設定"):
        st.session_state["fm_hz"] = 40.0

# スライダーでいつでも上書き可能
fm_hz = st.sidebar.slider(
    "変調周波数 FM (Hz)",
    0.5,
    100.0,
    key="fm_hz",
    help="2 Hz / 40 Hz が文献上よく用いられますが、任意の値に変更できます。",
)

# depth を離散メモリで指定（0.01〜0.10, 0.20, 0.30, 0.40, 0.50）
depth_options = [0.01, 0.02, 0.03, 0.04, 0.05,
                 0.06, 0.07, 0.08, 0.09, 0.10,
                 0.20, 0.30, 0.40, 0.50]

if "depth" not in st.session_state:
    # デフォルトはやや大きめ（0.30 = 30％）
    st.session_state["depth"] = 0.30

depth = st.sidebar.select_slider(
    "変調深度 depth（Δf/f）",
    options=depth_options,
    value=st.session_state["depth"],
    key="depth",
    help="キャリア周波数に対する揺れの割合です（例：0.02 = ±2％, 0.10 = ±10％, 0.50 = ±50％）。"
)

st.sidebar.markdown(
    """
**コメント**
- **0.01〜0.10**：1〜10％の揺れ（既報の閾値は多くがこの範囲）  
- **0.20〜0.50**：20〜50％の大きな揺れ（練習・重症例の確認用）  
- **2 Hz FM**：ゆっくりした高さの揺れ（プロソディ寄り）  
- **40 Hz FM**：粗い・ざらざらした高さの変動（音素レベルの変調寄り）
"""
)


def generate_fm_tone(
    sr: int,
    freq: float,
    dur_ms: int,
    fm_hz: float,
    depth: float,
    with_fm: bool,
    ear: str,
) -> bytes:
    """
    FMあり／なしの単音を生成して16-bit WAVバイト列を返す。
    ear: "両耳" / "左耳のみ" / "右耳のみ"
    """
    n_samples = int(sr * dur_ms / 1000)
    if n_samples <= 0:
        n_samples = 1
    t = np.arange(n_samples) / sr

    if with_fm and fm_hz > 0 and depth > 0:
        # 即時周波数：f(t) = f0 * (1 + depth * sin(2π fm t))
        inst_freq = freq * (1.0 + depth * np.sin(2.0 * np.pi * fm_hz * t))
        # phi[n] = phi[n-1] + 2π * f[n] / sr
        phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr
        tone = np.sin(phase)
    else:
        tone = np.sin(2.0 * np.pi * freq * t)

    # 短いHann窓で前後を丸めてクリックを軽減
    if n_samples > 3:
        window = np.hanning(n_samples)
        tone *= window

    # 正規化（最大振幅=1）
    max_abs = np.max(np.abs(tone))
    if max_abs < 1e-9:
        max_abs = 1.0
    tone = tone / max_abs

    # 16-bit PCM モノラル
    audio = (tone * 32767).astype(np.int16)

    # ステレオ化：耳条件に応じて L/R を振り分け
    if ear == "左耳のみ":
        left = audio
        right = np.zeros_like(audio)
    elif ear == "右耳のみ":
        left = np.zeros_like(audio)
        right = audio
    else:  # 両耳
        left = audio
        right = audio

    # L/Rをインターリーブしてステレオ配列に
    stereo = np.empty(2 * len(audio), dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)          # ステレオ
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())
    return buf.getvalue()


st.markdown("### ▶️ 刺激の再生")

st.write(
    dedent(
        f"""
        - **FMなし**：通常の純音（比較用）
        - **FMあり**：同じ周波数だが高さが揺れる音  
        - **ランダム**：FMあり／なしのどちらか一方をランダムに提示  

        現在のFM周波数設定：**{fm_hz:.1f} Hz**  
        現在の変調深度：**depth = {depth:.2f}（≈ ±{depth*100:.0f}％）**  
        出力チャネル：**{ear}**
        """
    )
)

col1, col2, col3 = st.columns(3)

if "last_random_label" not in st.session_state:
    st.session_state["last_random_label"] = "（まだ未実施）"

with col1:
    if st.button("🎵 FMなし（フラット）"):
        wav_bytes = generate_fm_tone(sr, freq, dur_ms, fm_hz, depth, with_fm=False, ear=ear)
        st.audio(wav_bytes, format="audio/wav", autoplay=True)

with col2:
    if st.button("🎵 FMあり（変調）"):
        wav_bytes = generate_fm_tone(sr, freq, dur_ms, fm_hz, depth, with_fm=True, ear=ear)
        st.audio(wav_bytes, format="audio/wav", autoplay=True)

with col3:
    if st.button("🎲 ランダム（一発）"):
        import random

        with_fm = bool(random.getrandbits(1))
        label = "FMあり" if with_fm else "FMなし"
        st.session_state["last_random_label"] = label
        wav_bytes = generate_fm_tone(sr, freq, dur_ms, fm_hz, depth, with_fm=with_fm, ear=ear)
        st.audio(wav_bytes, format="audio/wav", autoplay=True)

st.info(f"直近のランダム刺激：**{st.session_state['last_random_label']}**（検査者用メモ）")

st.markdown(
    """
---

### 🔎 推奨の使い方（例）

- **練習**：  
  まず depth = 0.30〜0.50 で「FMなし」「FMあり」を交互に聞かせて、  
  患者さんに「揺れている感じ」を体験してもらいます。  
  両耳 → 左耳のみ → 右耳のみ の順で聞き比べてもらうと、違和感の側を患者さん自身が報告しやすくなります。

- **閾値のざっくり推定**：  
  depth を 0.10 → 0.05 → 0.03 → 0.02 … と小さくしていき、  
  「一貫してFMありを区別できる最小のdepth」を耳別（左／右）にメモしておくと良いです。

- **2 Hz / 40 Hz × 耳別の比較**：  
  サイドバーの **2 Hz / 40 Hz ボタン**と「出力チャネル」を切り替え、  
  2 Hz / 40 Hz × 左耳 / 右耳 のそれぞれで必要な depth を比較すると、  
  片側の皮質聴覚障害やPPAサブタイプとの対応を検討しやすくなります。

※ 本アプリは **簡易スクリーニング／研究用プロトタイプ** です。  
  臨床での正式運用の際は、別途、手続き・スコアリング方法を標準化してください。
"""
)
