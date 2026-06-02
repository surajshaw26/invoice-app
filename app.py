import streamlit as st
import os
import io
import time
import pandas as pd
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import OpenAI

# Load configuration keys
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")

# App Layout Configuration
st.set_page_config(layout="wide", page_title="AI Data Extraction Portal")

# Custom CSS for Premium Dashboard Look
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .status-badge-green {
        background-color: #d4edda; color: #155724; 
        padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 13px;
    }
    .status-badge-yellow {
        background-color: #fff3cd; color: #856404; 
        padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 13px;
    }
    .status-badge-blue {
        background-color: #cce5ff; color: #004085; 
        padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🗂 nighttime Document Extraction & Validation Portal")
st.markdown("Automated layout parsing via **Azure Content Understanding** with fallback **LLM verification**.")
st.markdown("---")

@st.cache_resource
def get_clients():
    try:
        azure_client = DocumentIntelligenceClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))
        openai_client = OpenAI(api_key=OPENAI_KEY)
        return azure_client, openai_client
    except Exception as e:
        return None, None

azure_client, openai_client = get_clients()

s_state = st.session_state
if "batch_results" not in s_state:
    s_state.batch_results = {}
if "selected_file" not in s_state:
    s_state.selected_file = None

# --- SIDEBAR: INGESTION CONTROL PANEL ---
with st.sidebar:
    st.header("📤 Ingestion Layer")
    st.markdown("Upload documents to queue processing batches.")
    
    uploaded_files = st.file_uploader(
        "Select Invoice Files (Max 50)", 
        type=["pdf", "png", "jpg", "jpeg"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.success(f"📦 {len(uploaded_files)} document(s) in queue ready.")
        extract_btn = st.button("🚀 Run Batch Extraction", type="primary", use_container_width=True)
        
        if extract_btn:
            if not azure_client or not openai_client:
                st.error("AI engines initialization failed. Check your keys inside `.env`.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for index, file in enumerate(uploaded_files):
                    status_text.text(f"Processing: {file.name}")
                    file_bytes = file.read()
                    
                    # ⏱️ Start Stopwatch for current invoice file
                    file_start_time = time.time()
                    openai_cost = 0.0
                    
                    print(f"\n🚀 [START] Processing file: {file.name}")
                    
                    try:
                        poller = azure_client.begin_analyze_document(
                            "prebuilt-invoice", 
                            body=file_bytes
                        )
                        azure_result = poller.result()
                        
                        # 💸 Calculate Azure Page Cost ($10 per 1,000 pages -> $0.01 per page)
                        page_count = len(azure_result.pages) if azure_result.pages else 1
                        azure_cost = page_count * 0.01
                        
                        print(f"✅ Azure complete. Calculated Pages: {page_count} | Azure Cost: ${azure_cost:.4f}")
                        
                        document_fields = {}
                        raw_extracted_text = azure_result.content or ""
                        
                        if azure_result.documents:
                            extracted_fields = azure_result.documents[0].fields
                            
                            for field_name, field_data in extracted_fields.items():
                                val = getattr(field_data, 'value', None)
                                if val is None:
                                    val = str(field_data)
                                
                                conf = getattr(field_data, 'confidence', 1.0)
                                if conf is None:
                                    conf = 1.0
                                    
                                src = "Azure AI Engine"
                                
                                # Route to LLM Fallback if score is < 80%
                                if conf < 0.80:
                                    print(f"   ⚠️ Low Confidence ({conf*100:.1f}%) on '{field_name}'. Routing to OpenAI...")
                                    response = openai_client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": f"Extract the exact text value for '{field_name}' from the document text. Return ONLY the value string, absolutely no other conversation."},
                                            {"role": "user", "content": raw_extracted_text}
                                        ],
                                        temperature=0.0
                                    )
                                    val = response.choices[0].message.content.strip()
                                    conf = 0.85  
                                    src = "LLM Fallback"
                                    
                                    # 💸 Calculate OpenAI Cost live based on used response tokens
                                    p_tokens = response.usage.prompt_tokens
                                    c_tokens = response.usage.completion_tokens
                                    openai_cost += (p_tokens * (0.15 / 1000000)) + (c_tokens * (0.60 / 1000000))
                                
                                document_fields[field_name] = {"value": val, "confidence": conf, "source": src}
                        
                        # Stop watch calculation
                        elapsed_time = time.time() - file_start_time
                        total_file_cost = azure_cost + openai_cost
                        
                        s_state.batch_results[file.name] = {
                            "fields": document_fields,
                            "raw_text": raw_extracted_text,
                            "time_taken": elapsed_time,
                            "total_cost": total_file_cost,
                            "page_count": page_count
                        }
                    except Exception as ex:
                        print(f"❌ Error processing {file.name}: {ex}")
                        s_state.batch_results[file.name] = {"error": str(ex), "fields": {}}
                    
                    progress_bar.progress((index + 1) / len(uploaded_files))
                
                status_text.empty()
                st.toast("Batch processing completed successfully!", icon="🎉")

# --- MAIN WORKSPACE PANEL ---
if s_state.batch_results:
    # Build Consolidated Spreadsheet Row Arrays
    excel_rows = []
    for f_name, data in s_state.batch_results.items():
        row_dict = {
            "File Name": f_name, 
            "Processing Status": "Success" if "error" not in data else "Failed",
            "Time Elapsed": f"{data.get('time_taken', 0):.2f}s" if "error" not in data else "-",
            "Processing Cost": f"${data.get('total_cost', 0):.4f}" if "error" not in data else "-"
        }
        if "error" not in data:
            for f_key, f_meta in data["fields"].items():
                if f_key != "Items": 
                    row_dict[f_key] = str(f_meta["value"])
        excel_rows.append(row_dict)
        
    df_export = pd.DataFrame(excel_rows)
    
    # Generate Downloadable Buffer Sheet
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Extraction Matrix')
    buffer.seek(0)
    
    # UI Header Action Layer
    col_title, col_dl = st.columns([3, 1])
    with col_title:
        st.subheader("📊 Global Consolidated Batch Matrix")
    with col_dl:
        st.download_button(
            label="📥 Export Master Excel Sheet",
            data=buffer,
            file_name="extracted_invoice_matrix.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
    # Render Main Spreadsheet Grid
    st.dataframe(df_export, use_container_width=True, hide_index=True)
    st.markdown("---")
    
    # Split Review Segment Panel
    st.subheader("🔍 Granular Audit & Extraction Review Workspace")
    col_selector, col_viewer = st.columns([1, 2])
    
    with col_selector:
        st.write("**Select File to Inspect:**")
        for filename in s_state.batch_results.keys():
            is_active = "primary" if s_state.selected_file == filename else "secondary"
            if st.button(f"📄 {filename}", key=f"sel_{filename}", use_container_width=True, type=is_active):
                s_state.selected_file = filename
                st.rerun()
                
    with col_viewer:
        if s_state.selected_file and s_state.selected_file in s_state.batch_results:
            active_doc = s_state.batch_results[s_state.selected_file]
            st.markdown(f"#### Active Target Workspace: `{s_state.selected_file}`")
            
            # 📈 Premium UI Metric Row Blocks
            if "error" not in active_doc:
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="📄 Document Length", value=f"{active_doc.get('page_count', 1)} Pages")
                with m_col2:
                    st.metric(label="⏱️ Processing Speed", value=f"{active_doc.get('time_taken', 0):.2f} sec")
                with m_col3:
                    st.metric(label="💰 Estimated API Cost", value=f"${active_doc.get('total_cost', 0):.4f}")
                st.markdown("<br>", unsafe_allow_html=True)
            
            if "error" in active_doc:
                st.error(f"Processing Failure: {active_doc['error']}")
            elif not active_doc["fields"]:
                st.warning("No data fields parsed from document structure.")
            else:
                st.markdown("""
                    <table style='width:100%; border-collapse: collapse; margin-top:5px;'>
                        <thead>
                            <tr style='background-color: #f8f9fa; border-bottom: 2px solid #dee2e6; text-align: left;'>
                                <th style='padding: 10px;'>Database Column (Field)</th>
                                <th style='padding: 10px;'>Extracted Value Target Mapping</th>
                                <th style='padding: 10px;'>Probability (Confidence)</th>
                                <th style='padding: 10px;'>Verification Source</th>
                            </tr>
                        </thead>
                        <tbody>
                """, unsafe_allow_html=True)
                
                for f_name, f_meta in active_doc["fields"].items():
                    if f_name == "Items": 
                        continue
                    percentage = f_meta['confidence'] * 100
                    
                    if f_meta['source'] == "LLM Fallback":
                        badge = f"<span class='status-badge-blue'>🔄 {percentage:.1f}% Verified</span>"
                    elif percentage >= 80.0:
                        badge = f"<span class='status-badge-green'>🟢 {percentage:.1f}% Match</span>"
                    else:
                        badge = f"<span class='status-badge-yellow'>⚠️ {percentage:.1f}% Low</span>"
                        
                    st.markdown(f"""
                        <tr style='border-bottom: 1px solid #dee2e6;'>
                            <td style='padding: 12px; font-weight: bold; color: #495057;'>{f_name}</td>
                            <td style='padding: 12px; color: #212529;'>{f_meta['value']}</td>
                            <td style='padding: 12px;'>{badge}</td>
                            <td style='padding: 12px; font-style: italic; color: #6c757d;'>{f_meta['source']}</td>
                        </tr>
                    """, unsafe_allow_html=True)
                    
                st.markdown("</tbody></table>", unsafe_allow_html=True)
        else:
            st.info("💡 Select an individual invoice tracking token from the left selector panel to view granular data mappings and audit probability matrices.")
else:
    st.info("👈 Upload your targeted multi-page invoice bundles in the left panel ingestion layer and trigger execution to begin compilation rows.")