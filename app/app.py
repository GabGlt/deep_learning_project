import streamlit as st
import pandas as pd
from model_loader import load_model_and_tokenizer, predict

st.set_page_config(
    page_title="Deteksi Hate Speech",
    page_icon="🛡️",
    layout="wide",
)

ID2LABEL = {
    0: "Neutral Content",
    1: "Hate Speech / Abusive",
}

@st.cache_resource
def get_model():
    return load_model_and_tokenizer()

tokenizer, encoder, classifier = get_model()

st.markdown("""
<style>
/* HEADER */
.header {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    padding: 1.5rem;
    border-radius: 14px;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
}
/* CARDS */
.card {
    background-color: #111827;
    padding: 1.5rem;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    margin-top: 1rem;
    color: #f9fafb;
}
/* LABELS */
.label-aman { color: #22c55e; font-weight: 700; font-size: 22px; }
.label-bahaya { color: #ef4444; font-weight: 700; font-size: 22px; }
/* MARKED WORDS */
mark {
    background-color: #ffb3b3;
    padding: 0 2px;
    border-radius: 3px;
}
/* SIDEBAR */
.sidebar .stRadio > label {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Menu")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Single Text Classification", "Batch CSV Classification", "About"]
)

if menu == "Single Text Classification":
    st.markdown('<div class="header"><h1>🛡️ Deteksi Hate Speech</h1><p>Hate speech & abusive language (Bahasa Indonesia)</p></div>', unsafe_allow_html=True)

    with st.expander("📘 Cara Penggunaan", expanded=False):
        st.markdown(
            """
            1. Masukkan teks (komentar, tweet, dsb.) pada kotak teks.
            2. Klik tombol **Analisis**.
            3. Sistem menampilkan label, confidence, dan highlight kata bermasalah.
            """
        )

    text = st.text_area(
        "Masukkan teks yang ingin dianalisis:",
        height=160,
        placeholder="Contoh: Orang kayak gini emang bikin kesel, gak ada otaknya sama sekali..."
    )

    analyze = st.button("🔍 Analisis", use_container_width=True)

    if analyze:
        if not text.strip():
            st.warning("⚠️ Teks tidak boleh kosong.")
        else:
            with st.spinner("Menganalisis teks..."):
                label_id, probs = predict(text, tokenizer, encoder, classifier)

            label = ID2LABEL[label_id]
            confidence = probs[label_id]
            is_hate = label_id == 1

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"{'⚠️' if is_hate else '✅'} <span class='{'label-bahaya' if is_hate else 'label-aman'}'>{label}</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence:.2%}")
            st.divider()

            st.subheader("📊 Probabilitas Tiap Kelas")
            for i, label_name in ID2LABEL.items():
                st.write(label_name)
                st.progress(float(probs[i]))

            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("⚠️ Model dapat bias atau salah prediksi. Gunakan sebagai alat bantu.")

elif menu == "Batch CSV Classification":
    st.markdown('<div class="header"><h2>📂 Analisis Batch (CSV)</h2><p>Upload CSV untuk mendeteksi hate speech pada banyak teks sekaligus</p></div>', unsafe_allow_html=True)

    with st.expander("📘 Cara Penggunaan", expanded=False):
        st.markdown(
            """
            1. Siapkan file CSV dengan kolom **text** berisi komentar, tweet, atau teks lainnya.
            2. Upload file CSV melalui tombol di bawah.
            3. Sistem akan memproses setiap teks dan menampilkan label serta confidence.
            4. Download hasil prediksi sebagai file CSV jika diinginkan.
            """
        )

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV harus memiliki kolom bernama 'text'")
        else:
            results = []
            for t in df["text"].astype(str):
                label_id, probs = predict(t, tokenizer, encoder, classifier)
                results.append({"text": t, "label": ID2LABEL[label_id], "confidence": probs[label_id]})

            res_df = pd.DataFrame(results)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success("Analisis selesai")
            st.dataframe(res_df, use_container_width=True)

            csv = res_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Hasil", csv, "hasil_prediksi.csv", "text/csv")
            st.markdown('</div>', unsafe_allow_html=True)


elif menu == "About":
    st.markdown('<div class="header"><h2>Tentang Aplikasi</h2></div>', unsafe_allow_html=True)
    st.markdown("""
        **Hate Speech Detector**

        - Model: IndoBERT + SimCSE
        - Framework: PyTorch
        - UI & Deployment: Streamlit
        - Fitur:
            - Single text classification
            - Batch CSV prediction

        ⚠️ Model dapat bias dan tidak 100% akurat.
    """)
