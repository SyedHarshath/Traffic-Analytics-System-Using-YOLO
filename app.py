import streamlit as st
from PIL import Image
import tempfile
import os
from Traffic_Analytics_System_using_YOLO import predict

st.set_page_config(page_title="Traffic Analytics System",layout="wide")
st.title("Traffic Analytics System using YOLO")

uploaded_file = st.file_uploader("Upload a Road Traffic Image",type=["jpg","jpeg","png","webp","avif"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image,caption='Uploaded Image')
    file_extension = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False,suffix=file_extension) as tmp:
        tmp.write(uploaded_file.getvalue())
        image_path = tmp.name

    with st.spinner("Analyzing Traffic Scene..."):
        out_scores, out_boxes, out_classes, analysis, output_image = predict(image_path)

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image)
    
    with col2:
        st.subheader("Detection Result")
        st.image(output_image)

    st.subheader("Traffic Statistics")
    m1,m2,m3,m4,m5 = st.columns(5)

    m1.metric("Vehicles",analysis["vehicle_count"])

    m2.metric("Cars",analysis["vehicle_stats"]["car"])

    m3.metric("Buses",analysis["vehicle_stats"]["bus"])

    m4.metric("Trucks",analysis["vehicle_stats"]["truck"])

    m5.metric("Motorbikes",analysis["vehicle_stats"]["motorbike"])

    density = analysis["traffic_density"]

    if density == "LOW":
        st.success(f"Traffic Density: {density}")

    elif density == "MEDIUM":
        st.warning(f"Traffic Density: {density}")

    else:
        st.error(f"Traffic Density: {density}")
    
    st.subheader("Vehicle Distribution")

    vehicle_data = {
        "Car": analysis["vehicle_stats"]["car"],
        "Bus": analysis["vehicle_stats"]["bus"],
        "Truck": analysis["vehicle_stats"]["truck"],
        "Motorbike": analysis["vehicle_stats"]["motorbike"]
    }

    st.bar_chart(vehicle_data)