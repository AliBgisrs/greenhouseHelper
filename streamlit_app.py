import streamlit as st
import cv2
import numpy as np
import shapefile
import pandas as pd
import io
import zipfile

def get_shapefile_zip(contours, ppm):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        shp_io, shx_io, dbf_io = io.BytesIO(), io.BytesIO(), io.BytesIO()
        with shapefile.Writer(shp=shp_io, shx=shx_io, dbf=dbf_io) as w:
            w.field('Plant_ID', 'N')
            w.field('Area_cm2', 'F', decimal=3)
            for i, cnt in enumerate(contours):
                real_area = cv2.contourArea(cnt) / (ppm**2)
                # Convert contour to list of points
                w.poly([cnt.reshape(-1, 2).tolist()])
                w.record(i + 1, real_area)
        zf.writestr("canopy.shp", shp_io.getvalue())
        zf.writestr("canopy.shx", shx_io.getvalue())
        zf.writestr("canopy.dbf", dbf_io.getvalue())
    return buf.getvalue()

# App Configuration
st.set_page_config(layout="wide", page_title="Greenhouse Canopy Analyzer")

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    st.write("---")
    ppm = st.slider("Calibration (Pixels Per CM)", 10, 250, 85)
    min_size = st.number_input("Min Plant Size (Pixels)", value=500)
    st.write("---")
    # Your Branding
    st.markdown("### **Developed By Ali Bazrafkan**")
    st.info("Upload an image to start the automated canopy detection.")

# --- MAIN UI ---
st.title("🌿 Greenhouse Canopy: 3-Panel Analysis")

uploaded_file = st.file_uploader("Upload Greenhouse RGB Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Processing
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Generate Mask (Black & White)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    # Generate Overlay & Contours
    overlay_img = img_rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    stats = []
    
    for i, cnt in enumerate(contours):
        px_area = cv2.contourArea(cnt)
        if px_area > min_size:
            valid_contours.append(cnt)
            real_area = px_area / (ppm**2)
            
            # Draw on Overlay
            cv2.drawContours(overlay_img, [cnt], -1, (0, 255, 255), 3) # Yellow polygons
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.putText(overlay_img, str(i+1), (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            
            stats.append({"Plant ID": i+1, "Area (cm2)": round(real_area, 2)})

    # --- THREE PANEL DISPLAY ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Original Image")
        st.image(img_rgb, use_container_width=True)
        
    with col2:
        st.subheader("2. Detection Mask")
        st.image(mask, use_container_width=True)
        
    with col3:
        st.subheader("3. Result Overlay")
        st.image(overlay_img, use_container_width=True)

    # --- RESULTS TABLE & EXPORT ---
    st.divider()
    df_col, dl_col = st.columns([2, 1])
    
    with df_col:
        st.subheader("Individual Measurements")
        st.dataframe(pd.DataFrame(stats), use_container_width=True, height=300)
    
    with dl_col:
        st.subheader("Export Data")
        zip_data = get_shapefile_zip(valid_contours, ppm)
        st.download_button("📂 Download Shapefile (ZIP)", zip_data, "canopy_data.zip")
        st.download_button("📊 Download CSV Report", pd.DataFrame(stats).to_csv(index=False), "canopy_stats.csv")

else:

    st.warning("Please upload an image to view results.")
