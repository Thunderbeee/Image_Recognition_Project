import os
import json
import streamlit as st
from pathlib import Path
import shutil
import tempfile
from alpha_prototype import AlphaPrototype
from experiment_maker import ExperimentMaker

# st.set_page_config(page_title="Facial Recognition System", layout="wide")
# st.title("Facial Recognition System")

st.sidebar.header("Configuration")
include_public = st.sidebar.checkbox("Include Public Dataset", value=False)
model_options = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "ArcFace", "SFace"]
selected_model = st.sidebar.selectbox("Recognition Model", model_options)
distance_options = ["cosine", "euclidean"]
selected_distance = st.sidebar.selectbox("Distance Metric", distance_options)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.6)

def create_template_database():
    reference_db_path = "data/extracted"
    temp_template_path = "data/experiment/temp_templatedb.json"
    
    os.makedirs(os.path.dirname(temp_template_path), exist_ok=True)
    
    maker = ExperimentMaker(reference_db_path, include_student=True)
    
    all_individuals = []
    
    student_path = Path("student")
    if student_path.exists():
        for item in os.listdir(student_path):
            item_path = student_path / item
            if item_path.is_dir() and item not in maker.excluded_participants:
                all_individuals.append({"id": item, "path": item_path, "group": "student"})
    
    if include_public:
        for item in os.listdir(maker.reference_db_path):
            item_path = maker.reference_db_path / item
            if item_path.is_dir() and item not in maker.excluded_participants:
                all_individuals.append({"id": item, "path": item_path, "group": "public"})
    
    template_data = {}
    person_groups = {}
    
    for individual in all_individuals:
        individual_id = individual["id"]
        individual_path = individual["path"]
        individual_group = individual["group"]
        
        images = [f for f in os.listdir(individual_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()
        
        if len(images) >= 3:
            template_imgs = images[-3:]
        else:
            template_imgs = images
        
        template_data[individual_id] = [str(individual_path / img) for img in template_imgs]
        person_groups[individual_id] = individual_group
    
    with open(temp_template_path, 'w') as f:
        json.dump(template_data, f, indent=2)
    
    return temp_template_path, person_groups

st.header("Upload a photo for identification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        query_image_path = tmp_file.name
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, width=300)

    with st.spinner("Processing..."):
        template_path, person_groups = create_template_database()
        prototype = AlphaPrototype(template_path, selected_model, selected_distance)
        
        result = prototype.identify(query_image_path, threshold=confidence_threshold)
    
    with col2:
        st.subheader("Identification Result")
        
        if result["match_accepted"]:
            person_id = result["person_id"]
            group = person_groups.get(person_id, "unknown")
            if person_id == "200":
                st.success(f"They are {group} Weihao")
            elif person_id == "201":
                st.success(f"They are {group} Michelle")
            elif person_id == "202":
                st.success(f"They are {group} Anay")
            else:
                st.success(f"They are {group} {person_id}")
            
            st.write(f"Confidence: {1 - result['distance']:.2f}")
            
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            if person_id in template_data and len(template_data[person_id]) > 0:
                match_image_path = template_data[person_id][0]
                st.write("Best matching template image:")
                st.image(match_image_path, width=300)
        else:
            st.error("No match found")
    
    os.unlink(query_image_path)

st.markdown("---")
st.subheader("About the System")
st.write("""
This facial recognition system compares uploaded photos against a template database of known individuals.
It uses deep learning models to generate facial embeddings and calculate similarity.
""")
