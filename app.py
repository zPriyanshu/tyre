import streamlit as st
import os
import json
import pandas as pd
from PIL import Image
import sys

# Adjust path to import ocr_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_engine import TyreOCREngine, MLX_AVAILABLE

# Set page configuration
st.set_page_config(
    page_title="Tyre OCR Specification Extractor",
    page_icon="🛞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for styling
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    /* Main body background & fonts */
    .stApp {
        background-color: #0c0d16;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Title styling */
    .title-container {
        text-align: center;
        padding: 30px 10px;
        margin-bottom: 20px;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: -0.05em;
    }
    
    .sub-title {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 300;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Section Headings */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 8px;
    }
    
    /* Specification metrics list */
    .spec-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
    }
    
    .spec-item {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 14px 16px;
        transition: background 0.2s ease;
    }
    
    .spec-item:hover {
        background: rgba(99, 102, 241, 0.03);
    }
    
    .spec-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    
    .spec-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    
    .spec-value-highlight {
        color: #06b6d4;
    }
    
    /* Text bubble */
    .chip {
        display: inline-block;
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.2);
        color: #22d3ee;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    /* Spinner override for style */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
    
    /* File uploader styling overlay */
    section[data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        background: rgba(99, 102, 241, 0.01) !important;
        padding: 10px;
        transition: border-color 0.2s ease;
    }
    
    section[data-testid="stFileUploader"]:hover {
        border-color: #06b6d4 !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Main Title container
st.markdown(
    """
    <div class="title-container">
        <div class="main-title">🛞 TYRE OCR EXTRACTOR</div>
        <div class="sub-title">Extract tyre brand, size, load index, speed rating, DOT codes, and manufacturing history using cutting-edge Qwen-VL Vision Language Models.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar configurations
st.sidebar.markdown("### ⚙️ Engine Settings")
model_option = st.sidebar.selectbox(
    "Choose Model Size",
    ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"],
    help="Qwen2-VL-2B (~5.5GB download) is highly performant. Qwen2.5-VL-3B is the newer version."
)

use_mlx_opt = False
if MLX_AVAILABLE:
    use_mlx_opt = st.sidebar.checkbox(
        "Use MLX Acceleration",
        value=True,
        help="Runs optimized 4-bit model directly on Apple Silicon GPU for up to 3x speedups."
    )
else:
    st.sidebar.info("💡 MLX is not installed. Standard PyTorch (utilizing MPS/CUDA/CPU) will be used.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### ℹ️ How to read Tyre Sidewalls
    1. **Brand**: e.g., *Michelin*, *Bridgestone*, *Continental*.
    2. **Size**: e.g., *205/55R16* where:
       - **205** is width (mm)
       - **55** is aspect ratio (%)
       - **R** is radial construction
       - **16** is rim diameter (inches)
    3. **Load/Speed**: e.g., *91V* where:
       - **91** is load index (615 kg)
       - **V** is speed rating (240 km/h)
    4. **DOT Code**: e.g., *DOT XT 9E 1819*
       - Last 4 digits (*1819*) represent **Week 18, 2019**.
    """
)

# Function to cache model loading
@st.cache_resource(show_spinner=False)
def load_engine(model_id, use_mlx):
    engine = TyreOCREngine(model_id=model_id, use_mlx=use_mlx)
    engine.load_model()
    return engine

# Initialize engine loader state
engine = None
try:
    with st.spinner("Initializing neural engine (may take a moment on first load)..."):
        engine = load_engine(model_option, use_mlx_opt)
except Exception as e:
    st.error(f"Failed to load OCR model: {e}")
    st.info("Check if Hugging Face is accessible and you have sufficient RAM.")

# Create main interface divisions
col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-header">📸 Upload Tyre Sidewall Image</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader(
        "Select file...",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Style layout for the image preview
        st.image(image, use_column_width=True, caption="Source Image")
        
        # Save temp file for model ingestion
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        trigger_btn = st.button("🔍 Run Extraction", use_container_width=True, type="primary")
        
        if trigger_btn:
            if engine is not None:
                # Setup progress tracking
                progress_placeholder = st.empty()
                progress_placeholder.info("⚡ Extracting text and parsing markings using Qwen-VL...")
                
                try:
                    result = engine.extract_tyre_info(temp_path)
                    st.session_state["ocr_result"] = result
                    progress_placeholder.success("✅ Extraction complete!")
                except Exception as ex:
                    progress_placeholder.error(f"❌ Error during extraction: {ex}")
            else:
                st.warning("Neural engine could not be loaded. Check logs.")

with col_result:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-header">⚙️ Extracted Specifications</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if "ocr_result" in st.session_state:
        res = st.session_state["ocr_result"]
        
        # Layout metrics in structured HTML Grid
        specs_html = f"""
        <div class="glass-card">
            <div class="spec-grid">
                <div class="spec-item">
                    <div class="spec-label">Brand/Manufacturer</div>
                    <div class="spec-value spec-value-highlight">{res.get('brand') or 'Not Detected'}</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Model/Pattern Name</div>
                    <div class="spec-value">{res.get('model') or 'Not Detected'}</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Tyre Size</div>
                    <div class="spec-value spec-value-highlight">{res.get('tyre_size') or 'Not Detected'}</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Load & Speed rating</div>
                    <div class="spec-value">
                        {res.get('load_index') or ''}{res.get('speed_rating') or '' if res.get('load_index') or res.get('speed_rating') else 'Not Detected'}
                    </div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">DOT Code</div>
                    <div class="spec-value">{res.get('dot_code') or 'Not Detected'}</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Manufacturing Date</div>
                    <div class="spec-value">{res.get('manufacturing_date') or 'Not Detected'}</div>
                </div>
            </div>
            <div style="margin-top: 16px; padding: 12px; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.03); border-radius: 12px;">
                <div class="spec-label">Max Load & Inflation Pressure</div>
                <div class="spec-value" style="font-size: 0.95rem;">{res.get('max_load_pressure') or 'Not Detected'}</div>
            </div>
        </div>
        """
        st.markdown(specs_html, unsafe_allow_html=True)
        
        # Other markings chips section
        st.write("#### Special Markings")
        other_markings = res.get("other_markings", [])
        if other_markings:
            chips_html = "".join([f'<span class="chip">{m}</span>' for m in other_markings])
            st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
        else:
            st.info("No special markings detected.")
            
        # Full text dump
        with st.expander("📝 All Detected Sidewall Text"):
            for txt in res.get("all_sidewall_text", []):
                st.write(f"- {txt}")
                
        # Raw JSON section
        with st.expander("👾 Developer Output (Raw JSON)"):
            st.json(res)
            
        # Download results buttons
        csv_data = pd.DataFrame([res]).to_csv(index=False).encode('utf-8')
        json_data = json.dumps(res, indent=2).encode('utf-8')
        
        st.write("---")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📥 Download JSON Results",
                data=json_data,
                file_name="tyre_specs.json",
                mime="application/json",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="📥 Download CSV Summary",
                data=csv_data,
                file_name="tyre_specs.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px; border: 1px dashed rgba(255, 255, 255, 0.05); border-radius: 12px;">
                <span style="font-size: 3rem; display: block; margin-bottom: 10px;">🔍</span>
                <span style="color: #64748b; font-size: 1rem;">Upload an image and click <b>Run Extraction</b> to view specifications.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
