import streamlit as st
import pickle
from fpdf import FPDF

# Load vectorizer once
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Dictionary mapping model names to files
models = {
    "Logistic Regression": "LogisticRegression_model.pkl",
    "SVM": "SVM_model.pkl",
    "Random Forest": "RandomForest_model.pkl",
    "Decision Tree": "DecisionTree_model.pkl"
}

def generate_pdf(text, model_name, prediction, filename="result.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "Sentiment Analysis Result", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Model Used: {model_name}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Input Text:\n{text}")
    pdf.ln(5)
    pdf.cell(0, 10, f"Predicted Sentiment: {prediction}", ln=True)
    pdf.output(filename)
    return filename

st.title("Sentiment Analysis Deployment")

model_choice = st.selectbox("Select Model", list(models.keys()))
user_text = st.text_area("Enter text for sentiment analysis:")

if st.button("Predict"):
    # Load the selected model
    with open(models[model_choice], 'rb') as f:
        model = pickle.load(f)
    # Vectorize input text
    text_vec = vectorizer.transform([user_text.lower()])
    # Predict sentiment
    pred = model.predict(text_vec)[0]

    st.write(f"**Predicted Sentiment:** {pred}")

    # Generate PDF and enable download
    pdf_path = generate_pdf(user_text, model_choice, pred)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download Result as PDF",
            data=f,
            file_name=pdf_path,
            mime="application/pdf"
        )
