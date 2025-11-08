import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

# Streamlit page setup
st.set_page_config(page_title="AI Legal Translator", layout="wide")

# ✅ Law-themed background + blue content card
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=compress&w=1600");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.main-card {
    background: rgba(0, 77, 153, 0.85); /* College Blue with transparency */
    padding: 30px;
    border-radius: 15px;
    color: white;
}
h1, h2 {
    text-align: center;
    color: #ffffff;
    font-weight: 800;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ✅ Main content block
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1>Malnad College Of Engineering, Hassan</h1>", unsafe_allow_html=True)
st.markdown("<h2>⚖️ AI Enabled Regional Legal Translator</h2>", unsafe_allow_html=True)

# ✅ Team list (simple, no picture)
st.write("""
### ✅ Project By:
- Nithyashree CP  
- Samyuktha HS  
- Archana K  
- Avaneesh Honnappa
""")

# ✅ Load distilled model (works on Streamlit Cloud)
model_name = "ai4bharat/indictrans2-en-indic-distilled"
st.info("⏳ Loading translation model (one-time)")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
st.success("✅ Model Loaded")

# ✅ Input
st.subheader("Enter English Legal Text")
text = st.text_area("", height=160)

# ✅ Translate button
if st.button("Translate ✅"):
    if text.strip() == "":
        st.error("Please enter some text")
    else:
        formatted = f"eng_Latn kan_Knda {text}"
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, padding=True)
        outputs = model.generate(**inputs, max_length=256, num_beams=5)

        dev = tokenizer.decode(outputs[0], skip_special_tokens=True)
        kannada = UnicodeIndicTransliterator.transliterate(dev, "hi", "kn")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Devanagari Output")
            st.text_area("", dev, height=150)
        with c2:
            st.subheader("Kannada Output")
            st.text_area("", kannada, height=150)

st.markdown("</div>", unsafe_allow_html=True)
