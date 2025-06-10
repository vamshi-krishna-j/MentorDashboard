# mentor_dashboard.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
import os
from datetime import datetime
import uuid
from weasyprint import HTML
import tempfile
import markdown

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("topics.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS topics 
                 (id TEXT, topic TEXT, difficulty TEXT, pre_class TEXT, in_class TEXT, post_class TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Save generated content to SQLite
def save_to_db(topic, difficulty, pre_class, in_class, post_class):
    conn = sqlite3.connect("topics.db")
    c = conn.cursor()
    c.execute('''INSERT INTO topics (id, topic, difficulty, pre_class, in_class, post_class, timestamp) 
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (str(uuid.uuid4()), topic, difficulty, pre_class, in_class, post_class, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# Get Gemini LLM instance
def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API key not found. Please check your .env file.")
        return None
    return ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-pro")

# Generate PDF from Markdown
def generate_pdf(content, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        html_body = markdown.markdown(content)
        html_content = f"""
        <html>
            <head>
                <meta charset='utf-8'>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                    h1, h2, h3 {{ color: #2b6cb0; }}
                    ul {{ margin-left: 20px; }}
                </style>
            </head>
            <body>{html_body}</body>
        </html>
        """
        tmp.write(html_content.encode("utf-8"))
        tmp_path = tmp.name
    pdf_path = tmp_path.replace(".html", ".pdf")
    HTML(tmp_path).write_pdf(pdf_path)
    with open(pdf_path, "rb") as pdf_file:
        st.download_button("Download PDF", data=pdf_file.read(), file_name=filename, mime="application/pdf")

# Initialize DB
init_db()

# Streamlit UI setup
st.set_page_config(page_title="Mentor Dashboard", layout="centered")
st.title("Mentor Dashboard for Placement Prep")

if "step" not in st.session_state:
    st.session_state.step = "material_selection"
if "material_type" not in st.session_state:
    st.session_state.material_type = None

# Prompt templates
pre_class_prompt = PromptTemplate(
    input_variables=["topic", "difficulty"],
    template="""
You are an expert educator creating a pre-class document for a 1-hour placement prep class for {difficulty} level final-year undergraduate students at IIT Bombay.
Topic: {topic}

Generate a concise 1-2 page document (in Markdown format) to help students prepare. Include:
- A brief overview of the topic
- Key concepts and definitions
- 2-3 simple examples to illustrate core ideas
- 1-2 preparatory questions to think about
Ensure the content is clear, engaging, and suitable for the {difficulty} level.
"""
)

in_class_prompt = PromptTemplate(
    input_variables=["topic", "difficulty"],
    template="""
You are an expert educator creating an in-class lesson plan for a 1-hour placement prep class for {difficulty} level final-year undergraduate students at IIT Bombay.
Topic: {topic}

Generate a structured lesson plan (in Markdown format) for the mentor. Include:
- A 5-minute introduction
- A detailed flow of topics (with timings)
- At least 3 practical examples or problems to teach
- Key points to emphasize
- Suggestions for student engagement
"""
)

post_class_prompt = PromptTemplate(
    input_variables=["topic", "difficulty"],
    template="""
You are an expert educator creating a post-class document for a 1-hour placement prep class for {difficulty} level final-year undergraduate students at IIT Bombay.
Topic: {topic}

Generate a document (in Markdown format) that includes:
- A concise summary of key takeaways
- A quiz with 6-10 questions
- Answers to the quiz questions
"""
)

if st.session_state.step == "material_selection":
    with st.form("select_material_form"):
        topic = st.text_input("Enter Topic", placeholder="e.g., Dynamic Programming")
        difficulty = st.selectbox("Select Difficulty", ["Beginner", "Intermediate", "Advanced"])
        material_type = st.radio("Select Material Type", ["In-Class Document", "Pre-Class Document", "Post-Class Document"])
        submitted = st.form_submit_button("Generate Material")

        if submitted and topic:
            st.session_state.topic = topic
            st.session_state.difficulty = difficulty
            st.session_state.material_type = material_type
            st.session_state.step = "generate"

elif st.session_state.step == "generate":
    topic = st.session_state.topic
    difficulty = st.session_state.difficulty
    llm = get_llm()

    if llm:
        pre_class_chain = LLMChain(llm=llm, prompt=pre_class_prompt)
        in_class_chain = LLMChain(llm=llm, prompt=in_class_prompt)
        post_class_chain = LLMChain(llm=llm, prompt=post_class_prompt)

        pre_class_doc = pre_class_chain.run(topic=topic, difficulty=difficulty)
        in_class_doc = in_class_chain.run(topic=topic, difficulty=difficulty)
        post_class_doc = post_class_chain.run(topic=topic, difficulty=difficulty)

        save_to_db(topic, difficulty, pre_class_doc, in_class_doc, post_class_doc)

        st.subheader(f"{st.session_state.material_type}: {topic} ({difficulty})")

        if st.session_state.material_type == "Pre-Class Document":
            st.markdown(pre_class_doc)
            generate_pdf(pre_class_doc, f"pre_class_{topic}.pdf")
        elif st.session_state.material_type == "In-Class Document":
            st.markdown(in_class_doc)
            generate_pdf(in_class_doc, f"lesson_plan_{topic}.pdf")
        elif st.session_state.material_type == "Post-Class Document":
            st.markdown(post_class_doc)
            generate_pdf(post_class_doc, f"post_class_{topic}.pdf")

        if st.button("Generate Another Material"):
            st.session_state.step = "material_selection"
