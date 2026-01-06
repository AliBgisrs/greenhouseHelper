import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# --- Developed By Ali Bazrafkan ---

def analyze_greenhouse(image, ppm, eps_scale, min_samples):
    # 1. Image Pre-processing
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Refined green range for greenhouse lighting
    mask = cv2.inRange(hsv, np.array([30, 35, 35]), np.array([95, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    features = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                x, y, w, h = cv2.boundingRect(cnt)
                points.append({'contour': cnt, 'area': area, 'max_dim': max(w, h)})
                features.append([cx, cy])

    if not features:
        return mask, image, []

    # 2. Advanced DBSCAN Clustering
    X = np.array(features)
    # We use the average leaf size to define the search radius
    avg_leaf_size = np.median([p['max_dim'] for p in points])
    eps_value = avg_leaf_size * eps_scale
    
    clustering = DBSCAN(eps=eps_value, min_samples=int(min_samples)).fit(X)
    labels = clustering.labels_

    overlay = image.copy()
    pot_results = []
    
    # 3. Process Clusters and Sort Geographically
    unique_labels = set(labels)
    cluster_data = []

    for label in unique_labels:
        if label == -1: continue 
        
        indices = [i for i, l in enumerate(labels) if l == label]
        cluster_contours = [points[i]['contour'] for i in indices]
        all_pts = np.concatenate(cluster_contours)
        
        total_px_area = sum([points[i]['area'] for i in indices])
        hull = cv2.convexHull(all_pts)
        x, y, w, h = cv2.boundingRect(hull)
        
        # Store for sorting
        cluster_data.append({
            'hull': hull,
            'area_px': total_px_area,
            'center': (x + w//2, y + h//2),
            'bbox': (x, y, w, h)
        })

    # Sort Pots: First by Y (Row) then by X (Column)
    cluster_data = sorted(cluster_data, key=lambda p: (p['center'][1] // 50, p['center'][0]))

    for i, pot in enumerate(cluster_data):
        real_area = pot['area_px'] / (ppm**2)
        cv2.drawContours(overlay, [pot['hull']], -1, (0, 255, 0), 2)
        
        x, y, w, h = pot['bbox']
        cv2.putText(overlay, f"ID:{i+1}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        pot_results.append({"Pot ID": i+1, "Canopy Area (cm2)": round(real_area, 3)})

    return mask, overlay, pot_results

st.set_page_config(layout="wide", page_title="Ali Bazrafkan - Canopy Pro")
st.sidebar.title("App Settings")
st.sidebar.markdown("### **Developed By Ali Bazrafkan**")

with st.sidebar:
    ppm = st.slider("Scale (Pixels/CM)", 10, 200, 85)
    eps_scale = st.slider("Neighborhood Sensitivity", 0.5, 3.0, 1.2, 
                          help="Lower this if two separate plants are getting the same ID.")
    min_samples = st.number_input("Min Fragments per Pot", value=1)
    st.write("---")
    st.warning("If plants touch, lower 'Neighborhood Sensitivity' to split them.")

uploaded_file = st.file_uploader("Upload Greenhouse Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask, overlay, results = analyze_greenhouse(img_rgb, ppm, eps_scale, min_samples)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.image(img_rgb, caption="1. Original", use_container_width=True)
    with col2: st.image(mask, caption="2. Binary Mask", use_container_width=True)
    with col3: st.image(overlay, caption="3. Pot-by-Pot Analysis", use_container_width=True)

    if results:
        df = pd.DataFrame(results)
        st.subheader("Automated Canopy Inventory")
        st.dataframe(df, use_container_width=True)
        st.download_button("Export as CSV", df.to_csv(index=False), "greenhouse_data.csv")
