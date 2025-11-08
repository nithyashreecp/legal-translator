import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from huggingface_hub import login

# ✅ Login to HuggingFace for model download
login("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")

st.set_page_config(page_title="AI Legal Translator", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=compress&w=1600");
    background-size: cover;
    background-position: center;
}
.main-card {
    background: rgba(0, 77, 153, 0.85);
    padding: 30px;
    border-radius: 15px;
    color: white;
}
h1, h2 {
    text-align: center;
    color: #ffffff;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1>Malnad College Of Engineering, Hassan</h1>", unsafe_allow_html=True)
st.markdown("<h2>⚖️ AI Enabled Regional Legal Translator</h2>", unsafe_allow_html=True)

st.write("""
### ✅ Project By:
- Nithyashree CP  
- Samyuktha HS  
- Archana K  
- Avaneesh Honnappa
""")

model_name = "ai4bharat/indictrans2-en-indic-distilled"

st.info("⏳ Loading translation model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
st.success("✅ Model Loaded")

st.subheader("Enter English Legal Text")
text = st.text_area("", height=150)

if st.button("Translate ✅"):
    if text.strip() == "":
        st.error("Please type something.")
    else:
        formatted = f"eng_Latn kan_Knda {text}"
        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=256, num_beams=5)

        dev = tokenizer.decode(outputs[0], skip_special_tokens=True)
        kn = UnicodeIndicTransliterator.transliterate(dev, "hi", "kn")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Devanagari Output")
            st.text_area("", dev, height=150)
        with c2:
            st.subheader("Kannada Output")
            st.text_area("", kn, height=150)

st.markdown("</div>", unsafe_allow_html=True)

