import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

# Streamlit page setup
st.set_page_config(page_title="AI Legal Translator", layout="wide")

# Background CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1555374018-13a8994ab246?auto=compress&dpr=2&w=1600");
    background-size: cover;
    background-position: center;
}
.main-block {
    background: rgba(255,255,255,0.85);
    padding: 25px;
    border-radius: 12px;
}
h1, h2 {
    text-align: center;
    font-weight: 800;
    color: #002f6c;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title Section
st.markdown("<div class='main-block'>", unsafe_allow_html=True)
st.markdown("<h1>Malnad College Of Engineering, Hassan</h1>", unsafe_allow_html=True)
st.markdown("<h2>⚖️ AI Enabled Regional Legal Translator</h2>", unsafe_allow_html=True)

# Members
col1, col2 = st.columns([1,2])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/1048/1048953.png", width=200)
with col2:
    st.subheader("Project By:")
    st.write("""
    ✅ Nithyashree CP  
    ✅ Samyuktha HS  
    ✅ Archana K  
    ✅ Avaneesh Honnappa  
    """)

# Load model
model_name = "ai4bharat/indictrans2-en-indic-1B"
st.info("⏳ Loading translation model... (only first time)")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
st.success("✅ Model Ready!")

# Input box
st.subheader("Enter English Legal Text")
text = st.text_area("")

# Translate
if st.button("Translate ✅"):
    if text.strip() == "":
        st.error("Please enter some text first.")
    else:
        formatted = f"eng_Latn kan_Knda {text}"
        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=256, num_beams=5)

        # Devanagari → Kannada
        dev = tokenizer.decode(outputs[0], skip_special_tokens=True)
        kannada = UnicodeIndicTransliterator.transliterate(dev, "hi", "kn")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📝 Devanagari (Intermediate)")
            st.text_area("", dev, height=150)
        with c2:
            st.subheader("✅ Kannada Output")
            st.text_area("", kannada, height=150)

st.markdown("</div>", unsafe_allow_html=True)
