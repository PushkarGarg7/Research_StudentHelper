# app.py
import time
import json
import base64
import requests
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from flask import Flask, request
import threading
import webbrowser
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import io
from PyPDF2 import PdfReader
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
import openai
import tempfile
import pytesseract
import logging
import google.generativeai as genai
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage
from guardrails.hub import LlmRagEvaluator, HallucinationPrompt
import warnings
import logging
from guardrails.hub import NSFWText
import nltk
from rouge_score import rouge_scorer
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

nltk.download('punkt')

# Silence oauth2client warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set logging level for googleapiclient.discovery_cache
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)


# -------------------------------
# Configure Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file.")
    st.stop()

# openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# -------------------------------
# Constants
# -------------------------------
SCOPES = [
    'https://www.googleapis.com/auth/classroom.courses.readonly',
    'https://www.googleapis.com/auth/classroom.student-submissions.me.readonly',
    'https://www.googleapis.com/auth/classroom.announcements.readonly',
    'https://www.googleapis.com/auth/classroom.topics.readonly',
    'https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly',
    'https://www.googleapis.com/auth/classroom.coursework.students',
    'https://www.googleapis.com/auth/drive.readonly', # Added Drive Read-Only Scope
]

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update path if different
CLIENT_SECRET_FILE = 'client_secret_22403136260-23on12j9ip9948dhto3crga8sll5rh18.apps.googleusercontent.com.json'  # Ensure this file is correctly named and placed
PORT = 8081
REDIRECT_URI = f'http://localhost:{PORT}/'

# -------------------------------
# Initialize Flask app for OAuth callback
# -------------------------------
app = Flask(__name__)
auth_code = None
authorization_complete = threading.Event()

@app.route("/")
def oauth_callback():
    global auth_code
    auth_code = request.args.get('code')
    authorization_complete.set()
    return "Authorization successful! You can close this tab now."

def run_flask_app():
    app.run(port=PORT, debug=False, use_reloader=False)

# -------------------------------
# Helper Functions
# -------------------------------
def evaluate_response_metrics(reference, response):
    """
    Evaluates LLM response using ROUGE, BLEU, and Perplexity.
    """

    # ✅ Tokenization for BLEU
    reference_tokens = [nltk.word_tokenize(reference)]  # List of lists
    response_tokens = nltk.word_tokenize(response)

    # ✅ ROUGE Score with proper keys
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, response)

    # ✅ BLEU Score (using tokenized words)
    bleu_score = sacrebleu.corpus_bleu(
        [" ".join(response_tokens)],
        [[" ".join(ref) for ref in reference_tokens]]
    ).score

    # ✅ Perplexity Calculation (proper tokenization)
    ppl_model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(ppl_model_name)

    # Tokenize the response
    inputs = tokenizer(response, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits

    loss_fn = torch.nn.CrossEntropyLoss()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    # Compute loss and perplexity
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss).item()

    # ✅ Log Metrics
    metrics = {
        "ROUGE-1": rouge_scores["rouge1"].fmeasure,
        "ROUGE-2": rouge_scores["rouge2"].fmeasure,
        "ROUGE-L": rouge_scores["rougeL"].fmeasure,
        "BLEU": bleu_score,
        "Perplexity": perplexity
    }
    return metrics


def categorize_attachment(attachment):
    """
    Determines the type of attachment and returns its category.
    """
    if 'driveFile' in attachment:
        return 'driveFile'
    elif 'link' in attachment:
        return 'link'
    elif 'youtubeVideo' in attachment:
        return 'youtubeVideo'
    elif 'form' in attachment:
        return 'form'
    elif 'driveFolder' in attachment:
        return 'driveFolder'
    else:
        return 'unknown'

def fetch_pdf_content_from_drive(file_id, drive_service):
    """
    Downloads the PDF from Google Drive using the Drive API and saves it to a temporary file.
    Returns the file path of the saved file.
    """
    try:
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                st.info(f"Download {int(status.progress() * 100)}% complete.")

        fh.seek(0)
        # Save the PDF content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(fh.read())
            temp_pdf_path = temp_pdf.name

        return temp_pdf_path
    except HttpError as e:
        raise RuntimeError(f"Error downloading PDF from Drive: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error downloading PDF: {e}")
    
def preprocess_image_for_ocr(pil_image):
    """
    Enhances and sharpens the image for better OCR accuracy.
    """
    gray_image = pil_image.convert("L")
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.0)
    sharpened_image = enhanced_image.filter(ImageFilter.SHARPEN)
    return sharpened_image

def clean_ocr_text_preserve_lines(ocr_text):
    """
    Cleans OCR text while preserving line breaks.
    """
    lines = ocr_text.splitlines(keepends=False)
    cleaned_lines = [line.rstrip() for line in lines]
    return "\n".join(cleaned_lines)

def extract_text_and_images(pdf_path):
    """
    Extracts text and images from a PDF and returns combined text.
    """
    combined_output = []
    try:
        # Extract typed text using PyPDF2
        reader = PdfReader(pdf_path)
        typed_text_pages = [page.extract_text() for page in reader.pages]
    except Exception as e:
        st.error(f"Error reading typed text from PDF: {e}")
        typed_text_pages = []

    try:
        # Extract and OCR images using PyMuPDF
        doc = fitz.open(pdf_path)
        ocr_results = {}

        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    pil_image = preprocess_image_for_ocr(pil_image)
                    raw_ocr_text = pytesseract.image_to_string(pil_image, config="--psm 6")
                    ocr_text_cleaned = clean_ocr_text_preserve_lines(raw_ocr_text)
                    if ocr_text_cleaned.strip():
                        if page_index not in ocr_results:
                            ocr_results[page_index] = []
                        ocr_results[page_index].append(ocr_text_cleaned)
                except Exception as e:
                    st.warning(f"Error extracting image {img_index + 1} on page {page_index + 1}: {e}")
        doc.close()
    except Exception as e:
        st.error(f"Error opening PDF with fitz: {e}")
        ocr_results = {}

    # Combine typed text and OCR results
    for i, typed_text in enumerate(typed_text_pages):
        page_num = i + 1
        if typed_text and typed_text.strip():
            combined_output.append(f"=== Page {page_num} (Typed Text) ===\n{typed_text.strip()}")
        if i in ocr_results:
            for idx, text_block in enumerate(ocr_results[i], start=1):
                combined_output.append(f"=== Page {page_num} (OCR from Embedded Image {idx}) ===\n{text_block}")

    return "\n\n".join(combined_output)

def print_guard_report(text, result):
    """
    Prints detailed guard validation report.
    """
    print("\n=== Guard Validation Report ===")
    print(f"Text analyzed: {text[:100]}...")  # First 100 chars
    
    try:
        # Try to parse and format the validation result
        if hasattr(result, 'validation_response'):
            response = result.validation_response
            if isinstance(response, str):
                response = json.loads(response)
            print("\nValidation Details:")
            print(json.dumps(response, indent=2))
        else:
            print("Text passed validation")
    except Exception as e:
        print(f"Error parsing validation result: {e}")
    
    print("===========================\n")

def grade_submission(question_text, submission_text):
    """
    Grades the submission by first validating the submission text for toxicity,
    then generating an AI response, and finally computing response metrics.
    """
    # 1. Validate the submission text using ToxicLanguage
    toxic_guard = Guard().use(
        ToxicLanguage(
            threshold=1,
            validation_method="sentence",
            on_fail=OnFailAction.EXCEPTION
        )
    )
    try:
        print("\n🔍 Validating submission text for toxicity...")
        toxic_guard.validate(submission_text)
        print("✅ Submission passed toxicity check.")
    except Exception as e:
        print("❌ Submission toxicity validation failed:", e)
        return None

    # 2. Generate the LLM response
    prompt = (
        "You are an expert AI grader assigned to evaluate a student's submission "
        "based on a set of assignment questions. Your analysis must be precise, "
        "consistent, and structured. Assume that the submission may include errors "
        "from OCR (optical character recognition) and some parts of the content, "
        "especially text within images, may be missing or distorted. Grade based "
        "only on the provided text, without assuming or inferring missing information.\n\n"
        "Your grading report must include the following components:\n"
        "1. **Overall Grade (out of 10):** Provide a clear numeric score based on the quality, "
        "completeness, and alignment of the submission with the assignment questions.\n"
        "2. **Detailed Strengths:** Highlight specific strengths of the submission, with "
        "examples or references to relevant sections.\n"
        "3. **Detailed Areas for Improvement:** Identify and explain the key areas where the "
        "submission falls short. Provide constructive feedback with examples.\n"
        "4. **Question-by-Question Comments:** For each question in the assignment, evaluate "
        "how well the student addressed it. Include specific examples from the submission.\n\n"
        f"**Assignment Questions:**\n{question_text}\n\n"
        f"**Student Submission:**\n{submission_text}\n\n"
        "Ensure your feedback is professional, detailed, and helps the student understand "
        "how to improve. Use clear language and avoid ambiguity."
    )
    start_time = time.time()
    print("\n🤖 Generating AI response...")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    response_text = response.text
    end_time = time.time()

    # 3. Compute response metrics (time, length, ROUGE, BLEU, Perplexity)
    response_time = round(end_time - start_time, 2)
    response_length = len(response_text.split())
    metrics = evaluate_response_metrics(submission_text, response_text)
    metrics.update({
        "Response Time (s)": response_time,
        "Response Length (words)": response_length,
    })
    print("\n📊 LLM Evaluation Metrics:", metrics)

    return response_text, metrics

def update_grade(service, course_id, coursework_id, submission_id, grade):
    try:
        grade_float = float(grade)
        if grade_float < 0:
            raise ValueError("Grade cannot be negative")

        student_submission = {
            'assignedGrade': grade_float,
            'draftGrade': grade_float,  # Transition the submission state to RETURNED
        }

        # Define the update mask
        update_mask = 'assignedGrade,draftGrade'

        # Perform a single patch call to update grade and state
        service.courses().courseWork().studentSubmissions().patch(
            courseId=course_id,
            courseWorkId=coursework_id,
            id=submission_id,
            updateMask=update_mask,
            body=student_submission
        ).execute()
        return True, f"Grade {grade_float} successfully updated and returned to student."

    except ValueError as e:
        return False, f"Invalid grade format: {str(e)}"
    except HttpError as e:
        error_details = ""
        if hasattr(e, 'resp') and hasattr(e, 'content'):
            error_details = f"Status: {e.resp.status}, Content: {e.content.decode('utf-8')}"
        else:
            error_details = str(e)
        return False, f"Google Classroom API error: {error_details}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def fetch_courses_data(service):
    """
    Fetches courses, assignments, submissions, and announcements from Google Classroom.
    """
    all_data = []
    try:
        courses_result = service.courses().list(pageSize=100).execute()
        courses = courses_result.get('courses', [])

        if not courses:
            st.warning('No courses found.')
            return []

        for course in courses:
            course_data = {
                "name": course.get('name', 'No name'),
                "section": course.get('section', 'No section'),
                "id": course.get('id', 'No ID'),
                "assignments": [],
                "materials": [],
                "announcements": [],
                "messages": []
            }

            try:
                # Fetch coursework
                coursework_result = service.courses().courseWork().list(
                    courseId=course['id'], pageSize=100
                ).execute()
                coursework = coursework_result.get('courseWork', [])

                # Handle pagination for coursework
                while 'nextPageToken' in coursework_result:
                    coursework_result = service.courses().courseWork().list(
                        courseId=course['id'],
                        pageSize=100,
                        pageToken=coursework_result['nextPageToken']
                    ).execute()
                    coursework.extend(coursework_result.get('courseWork', []))

                for assignment in coursework:
                    assignment_data = {
                        "id": assignment.get('id'),
                        "title": assignment.get('title', 'Untitled'),
                        "due_date": format_deadline(
                            assignment.get('dueDate'),
                            assignment.get('dueTime')
                        ),
                        "work_type": assignment.get('workType', 'Not specified'),
                        "max_points": assignment.get('maxPoints', 'Not specified'),
                        "submissions": [],
                        "materials": []
                    }

                    # Process materials
                    for material in assignment.get('materials', []):
                        assignment_data["materials"].append(extract_material_info(material))

                    # Fetch submissions
                    try:
                        submissions_result = service.courses().courseWork().studentSubmissions().list(
                            courseId=course['id'],
                            courseWorkId=assignment['id'],
                            pageSize=100
                        ).execute()
                        submissions = submissions_result.get('studentSubmissions', [])

                        # Handle pagination for submissions
                        while 'nextPageToken' in submissions_result:
                            submissions_result = service.courses().courseWork().studentSubmissions().list(
                                courseId=course['id'],
                                courseWorkId=assignment['id'],
                                pageSize=100,
                                pageToken=submissions_result['nextPageToken']
                            ).execute()
                            submissions.extend(submissions_result.get('studentSubmissions', []))

                        for submission in submissions:
                            submission_data = {
                                "id": submission.get('id'),
                                "user_id": submission.get('userId', 'No user ID'),
                                "state": submission.get('state', 'Unknown'),
                                "assigned_grade": submission.get('assignedGrade', 'Not graded'),
                                "grade": submission.get('grade', 'Not graded'),
                                "late": submission.get('late', 'Unknown'),
                                "submission_time": submission.get('updateTime', 'No submission time'),
                                "alternate_link": submission.get('alternateLink', 'No Link'),
                                "attachments": []
                            }

                            # Process attachments
                            if 'assignmentSubmission' in submission:
                                assignment_submission = submission['assignmentSubmission']
                                if 'attachments' in assignment_submission:
                                    for attachment in assignment_submission['attachments']:
                                        attachment_data = {}
                                        if 'driveFile' in attachment:
                                            attachment_data = {
                                                "file_name": attachment['driveFile'].get('title', 'No title'),
                                                "file_url": attachment['driveFile'].get('alternateLink', 'No URL'),
                                                "driveFile": attachment['driveFile']
                                            }
                                        elif 'link' in attachment:
                                            attachment_data = {
                                                "file_name": attachment['link'].get('title', 'No title'),
                                                "file_url": attachment['link'].get('url', 'No URL'),
                                                "link": attachment['link']
                                            }
                                        elif 'youtubeVideo' in attachment:
                                            attachment_data = {
                                                "file_name": "YouTube Video",
                                                "file_url": attachment['youtubeVideo'].get('url', 'No URL'),
                                                "youtubeVideo": attachment['youtubeVideo']
                                            }
                                            submission_data["attachments"].append(attachment_data)
                                        elif 'form' in attachment:
                                            attachment_data = {
                                                "file_name": attachment['form'].get('title', 'No title'),
                                                "file_url": attachment['form'].get('formUrl', 'No URL'),
                                                "form": attachment['form']
                                            }
                                            submission_data["attachments"].append(attachment_data)
                                        elif 'driveFolder' in attachment:
                                            attachment_data = {
                                                "file_name": attachment['driveFolder'].get('title', 'No title'),
                                                "file_url": attachment['driveFolder'].get('url', 'No URL'),
                                                "driveFolder": attachment['driveFolder']
                                            }
                                            submission_data["attachments"].append(attachment_data)
                                        else:
                                            attachment_data = {
                                                "file_name": "Unknown Attachment",
                                                "file_url": "#",
                                                "unknown": attachment
                                            }
                                        
                                        if attachment_data:
                                            submission_data["attachments"].append(attachment_data)

                            assignment_data["submissions"].append(submission_data)

                    except HttpError as e:
                        st.error(f"Error fetching submissions for assignment {assignment.get('title', 'Unknown')}: {e}")

                    course_data["assignments"].append(assignment_data)

            except HttpError as e:
                st.error(f"Error fetching assignments for course {course.get('name', 'Unknown')}: {e}")

            try:
                # Fetch announcements
                announcements_result = service.courses().announcements().list(
                    courseId=course['id'], pageSize=100
                ).execute()
                announcements = announcements_result.get('announcements', [])

                # Handle pagination for announcements
                while 'nextPageToken' in announcements_result:
                    announcements_result = service.courses().announcements().list(
                        courseId=course['id'],
                        pageSize=100,
                        pageToken=announcements_result['nextPageToken']
                    ).execute()
                    announcements.extend(announcements_result.get('announcements', []))

                for announcement in announcements:
                    course_data["messages"].append(announcement.get('text', 'No content'))

            except HttpError as e:
                st.error(f"Error fetching announcements for course {course.get('name', 'Unknown')}: {e}")

            all_data.append(course_data)

        return all_data

    except Exception as e:
        st.error(f"Error fetching courses: {e}")
        raise  # Re-raise the exception to help with debugging

    
def extract_material_info(material):
    if 'link' in material:
        return {
            "title": material['link'].get('title', 'No title'),
            "url": material['link'].get('url', 'No URL'),
            "description": material['link'].get('description', 'No description')
        }
    elif 'driveFile' in material:
        return {
            "title": material['driveFile']['driveFile'].get('title', 'No title'),
            "url": material['driveFile']['driveFile'].get('alternateLink', 'No URL'),
            "description": material['driveFile']['driveFile'].get('description', 'No description')
        }
    elif 'youtubeVideo' in material:
        return {
            "title": material['youtubeVideo'].get('title', 'No title'),
            "url": material['youtubeVideo'].get('alternateLink', 'No URL'),
            "description": material['youtubeVideo'].get('description', 'No description')
        }
    elif 'form' in material:
        return {
            "title": material['form'].get('title', 'No title'),
            "url": material['form'].get('formUrl', 'No URL'),
            "description": material['form'].get('thumbnailUrl', 'No description')
        }
    return {"title": "Unknown Material Type", "url": "No URL", "description": "N/A"}

def format_deadline(due_date = None, due_time = None):
    if not due_date:
        return "No deadline set"

    utc_zone = pytz.utc
    ist_zone = pytz.timezone("Asia/Kolkata")

    year = due_date.get('year', '')
    month = str(due_date.get('month', '')).zfill(2)
    day = str(due_date.get('day', '')).zfill(2)

    time_str = ''
    if due_time:
        hours = str(due_time.get('hours', '00')).zfill(2)
        minutes = str(due_time.get('minutes', '00')).zfill(2)
        time_str = f" {hours}:{minutes}"

    datetime_str = f"{year}-{month}-{day}{time_str}"
    try:
        utc_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
        utc_time = utc_zone.localize(utc_time)
        ist_time = utc_time.astimezone(ist_zone)
        return ist_time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        return f"Error in converting time: {str(e)}"

def create_classroom_documents(courses_data):
    """
    Creates Document objects from the fetched courses data for chatbot ingestion.
    """
    documents = []
    for course in courses_data:
        course_content = []
        course_name = course.get('name', 'No name')
        course_id = course.get('id', 'No ID')

        # Add course basic info
        course_content.append(f"Course Name: {course_name}")
        course_content.append(f"Section: {course.get('section', 'No section')}")

        # Add assignments
        course_content.append("Assignments:")
        for assignment in course.get('assignments', []):
            course_content.extend([
                f"  - Title: {assignment.get('title', 'No title')}",
                f"    Due Date: {assignment.get('due_date', 'No due date')}",
                f"    Work Type: {assignment.get('work_type', 'No work type')}",
                f"    Max Points: {assignment.get('max_points', 'No max points')}"
            ])

            # Add materials
            if assignment.get("materials"):
                course_content.append("    Materials:")
                for material in assignment["materials"]:
                    course_content.extend([
                        f"      * Title: {material.get('title', 'No title')}",
                        f"        URL: {material.get('url', 'No URL')}"
                    ])

            # Add submissions with grades
            course_content.append("    Submissions:")
            for submission in assignment.get("submissions", []):
                course_content.extend([
                    f"      * User ID: {submission.get('user_id', 'No user ID')}",
                    f"        State: {submission.get('state', 'No state')}",
                    f"        Grade: {submission.get('grade', 'Not graded')}",
                    f"        Assigned Grade: {submission.get('assigned_grade', 'Not graded')}",
                    f"        Submission Time: {submission.get('submission_time', 'No submission time')}",
                    f"        Late: {submission.get('late', 'Unknown')}"
                ])
                if submission.get("attachments"):
                    course_content.append("        Attachments:")
                    for attachment in submission["attachments"]:
                        if 'driveFile' in attachment:
                            course_content.append(
                                f"          - {attachment['file_name']}: {attachment['file_url']}"
                            )
                        elif 'link' in attachment:
                            course_content.append(
                                f"          - Link: {attachment['file_url']} ({attachment['file_name']})"
                            )
                        elif 'youtubeVideo' in attachment:
                            course_content.append(
                                f"          - YouTube Video: {attachment['file_url']}"
                            )
                        elif 'form' in attachment:
                            course_content.append(
                                f"          - Form: {attachment['file_url']} ({attachment['file_name']})"
                            )
                        elif 'driveFolder' in attachment:
                            course_content.append(
                                f"          - Drive Folder: {attachment['file_url']} ({attachment['file_name']})"
                            )
                        else:
                            course_content.append(
                                f"          - Unknown Attachment Type: {attachment.get('file_url', 'No URL')}"
                            )

        # Add messages/announcements
        course_content.append("\nMessages:")
        for message in course.get("messages", []):
            course_content.append(f"  - {message}")

        # Create a Document object with course metadata
        doc = Document(
            page_content="\n".join(course_content),
            metadata={
                "course_name": course_name,
                "course_id": course_id,
                "doc_type": "classroom_course"
            }
        )
        documents.append(doc)

    return documents

def setup_chatbot(documents, timestamp):
    """
    Sets up the chatbot using the ingested documents.
    """
    try:
        persist_directory = f"doc_db_{timestamp}"
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        llm = ChatGroq(
            model="llama-3.2-90b-vision-preview",
            temperature=0
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return vectorstore, qa_chain
    except Exception as e:
        st.error(f"Error setting up chatbot: {e}")
        return None, None

import base64

# -------------------------------
# Student Interface
# -------------------------------
def display_student_interface(st):
    # Display Courses and Assignments
    if st.session_state.courses_data:
        st.header("📂 Your Courses")
        for course in st.session_state.courses_data:
            with st.expander(f"**{course['name']}** (Section: {course['section']})"):
                st.subheader("📋 Assignments")
                if course['assignments']:
                    for assignment in course['assignments']:
                        st.markdown(f"### {assignment['title']}")
                        st.write(f"**Due Date:** {assignment['due_date']}")
                        st.write(f"**Work Type:** {assignment['work_type']}")
                        st.write(f"**Max Points:** {assignment['max_points']}")

                        # Materials Section
                        if assignment['materials']:
                            st.markdown("**📚 Materials:**")
                            for material in assignment['materials']:
                                st.markdown(f"- [{material['title']}]({material['url']})")
                                st.write(f"  * {material['description']}")

                        # Submissions Section
                        if assignment['submissions']:
                            st.markdown("**📝 Submissions:**")
                            for submission in assignment['submissions']:
                                st.markdown(f"- **User ID:** {submission.get('user_id', 'N/A')}")
                                st.write(f"  - **State:** {submission.get('state', 'N/A')}")
                                st.write(f"  - **Assigned Grade:** {submission.get('assigned_grade', 'N/A')}")
                                st.write(f"  - **Grade:** {submission.get('grade', 'N/A')}")
                                st.write(f"  - **Late:** {submission.get('late', 'N/A')}")
                                st.write(f"  - **Submission Time:** {submission.get('submission_time', 'N/A')}")
                                if submission.get("attachments"):
                                    st.markdown("  - **Attachments:**")
                                    for attachment in submission["attachments"]:
                                        attachment_type = categorize_attachment(attachment)
                                        if attachment_type == 'driveFile':
                                            st.markdown(
                                                f"    - [{attachment['file_name']}]({attachment['file_url']})"
                                            )
                                        elif attachment_type == 'link':
                                            st.markdown(
                                                f"    - [Link: {attachment['file_name']}]({attachment['file_url']})"
                                            )
                                        elif attachment_type == 'youtubeVideo':
                                            st.markdown(
                                                f"    - [YouTube Video]({attachment['file_url']})"
                                            )
                                        elif attachment_type == 'form':
                                            st.markdown(
                                                f"    - [Form: {attachment['file_name']}]({attachment['file_url']})"
                                            )
                                        elif attachment_type == 'driveFolder':
                                            st.markdown(
                                                f"    - [Drive Folder: {attachment['file_name']}]({attachment['file_url']})"
                                            )
                                        else:
                                            st.markdown(
                                                f"    - [Unknown Attachment]({attachment.get('file_url', '#')})"
                                            )
                else:
                    st.write("No assignments found.")

                # Announcements Section
                st.subheader("📢 Announcements")
                if course['messages']:
                    for message in course['messages']:
                        st.write(f"- {message}")
                else:
                    st.write("No announcements found.")

    # Chatbot Interaction
    guard = Guard().use(
        ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
    )
    guard2 = Guard().use(
        LlmRagEvaluator(
            eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
            llm_evaluator_fail_response="hallucinated",
            llm_evaluator_pass_response="factual",
            llm_callable="gemini/gemini-1.5-flash",
            on_fail="exception",
            on="prompt"
        ),
    )
    if st.session_state.chatbot_ready:
        st.header("💬 Chatbot")
        
        # Add a toggle for showing metrics
        show_metrics = st.checkbox("Show Response Metrics", value=False)
        
        user_query = st.text_input("Ask your question about your courses:", "")
        
        if st.button("🔍 Submit"):
            if user_query.strip() == "":
                st.warning("Please enter a valid question.")
            else:
                try:
                    # Validate the user query using the guard
                    guard.validate(user_query)
                except Exception as guard_error:
                    st.error(f"Query validation failed: {guard_error}")
                else:
                    with st.spinner("Processing your query..."):
                        try:
                            response = st.session_state.qa_chain.invoke({"query": user_query})
                            
                            # Extract context from source documents
                            context_text = " ".join(doc.page_content for doc in response.get('source_documents', []))

                            st.markdown("**Answer:**")
                            st.write(response['result'])

                            # Display source documents in expandable section
                            with st.expander("📝 Source Documents"):
                                for idx, doc in enumerate(response['source_documents']):
                                    st.markdown(f"**Document {idx + 1}:**")
                                    st.write(doc.page_content)
                                    st.markdown("---")
                            
                            # Limit context to 1000 characters
                            max_context_length = 10000 
                            trimmed_context = context_text[:max_context_length] + "..." if len(context_text) > max_context_length else context_text

                            metadata = {
                                "user_message": user_query,
                                "context": {"retrieved_docs": trimmed_context},  # Ensure it's within a reasonable length
                                "llm_response": response['result']
                            }

                            validation_result = guard2.validate(llm_output=response['result'], metadata=metadata)

                            st.markdown("**Validation Result:**")
                            st.write(validation_result)  # This will show whether the response is factual or hallucinated


                            
                        except Exception as e:
                            st.error(f"Error processing query: {e}")

                        
# -------------------------------
# Teacher Interface
# -------------------------------
def display_teacher_interface(st, service, drive_service):
    st.header("👩‍🏫 Your Courses and Assignments")
    for course in st.session_state.courses_data:
        with st.expander(f"**{course['name']}** (Section: {course['section']})", expanded=False):
            st.subheader("📋 Assignments")
            if course['assignments']:
                for assignment in course['assignments']:
                    st.markdown(f"### {assignment['title']}")
                    st.write(f"**Due Date:** {assignment['due_date']}")
                    st.write(f"**Work Type:** {assignment['work_type']}")
                    st.write(f"**Max Points:** {assignment['max_points']}")
    
                    # Materials Section
                    if assignment.get('materials'):
                        st.markdown("**📚 Materials:**")
                        for material in assignment["materials"]:
                            st.markdown(f"- [{material['title']}]({material['url']})")
                            st.write(f"  * {material['description']}")
    
                    # Submissions Section
                    if assignment.get('submissions'):
                        st.markdown("**📝 Submissions:**")
                        for submission in assignment['submissions']:
                            # Safely get submission ID with fallback
                            submission_id = submission.get('id')
                            if not submission_id:
                                st.error("Invalid submission: Missing submission ID")
                                continue
    
                            # Create unique keys for session state
                            grading_key = f"grading_{submission_id}"
                            report_key = f"report_{submission_id}"
    
                            # Initialize session state for this submission if not exists
                            if grading_key not in st.session_state:
                                st.session_state[grading_key] = False
                            if report_key not in st.session_state:
                                st.session_state[report_key] = None
    
                            # Display submission details
                            st.markdown(
                                f"- **Submission ID:** {submission_id} | **User ID:** {submission.get('user_id', 'N/A')}"
                            )
                            st.write(f"  - **State:** {submission.get('state', 'N/A')}")
                            st.write(f"  - **Assigned Grade:** {submission.get('assigned_grade', 'N/A')}")
                            st.write(f"  - **Grade:** {submission.get('grade', 'N/A')}")
                            st.write(f"  - **Late:** {submission.get('late', 'N/A')}")
                            st.write(f"  - **Submission Time:** {submission.get('submission_time', 'N/A')}")
    
                            # Display attachments
                            attachments = submission.get("attachments", [])
                            if attachments:
                                st.markdown("  - **Attachments:**")
                                for attachment in attachments:
                                    try:
                                        attachment_type = categorize_attachment(attachment)
                                        if attachment_type == 'driveFile':
                                            file_name = attachment.get('file_name', 'Unnamed File')
                                            file_url = attachment.get('file_url', '#')
                                            st.markdown(f"    - [{file_name}]({file_url})")
                                    except Exception as e:
                                        st.warning(f"Error displaying attachment: {str(e)}")
    
                            # Grading section
                            if submission.get('state') != 'RETURNED':
                                col1, col2 = st.columns([1, 3])
    
                                # Start grading button
                                if not st.session_state[grading_key]:
                                    if col1.button("Start Grading", key=f"start_grade_{submission_id}"):
                                        st.session_state[grading_key] = True
    
                                # Grading interface
                                if st.session_state[grading_key]:
                                    try:
                                        # Only generate report if not already generated
                                        if st.session_state[report_key] is None:
                                            with st.spinner("Analyzing submission..."):
                                                # Extract assignment content
                                                materials = assignment.get("materials", [])
                                                question_text = "\n".join([
                                                    f"{material.get('title', '')}: {material.get('description', '')}\n{material.get('url', '')}"
                                                    for material in materials
                                                ])
    
                                                # Extract submission content
                                                submission_text = ""
                                                for attachment in attachments:
                                                    try:
                                                        if categorize_attachment(attachment) == 'driveFile':
                                                            drive_file = attachment.get('driveFile', {})
                                                            file_id = drive_file.get('id')
                                                            if file_id:
                                                                pdf_path = fetch_pdf_content_from_drive(file_id, drive_service)
                                                                submission_text += extract_text_and_images(pdf_path)
                                                                os.unlink(pdf_path)
                                                            else:
                                                                st.warning("Missing file ID in drive attachment")
                                                    except Exception as e:
                                                        st.error(f"Error processing attachment: {str(e)}")
    
                                                if submission_text.strip():
                                                    st.session_state[report_key] = grade_submission(question_text, submission_text)
                                                else:
                                                    st.warning("No text extracted from submission.")
                                                    st.session_state[grading_key] = False
                                                    continue
    
                                        # Display grading report using markdown
                                        if st.session_state[report_key]:
                                            # Use a container to organize the grading report and actions
                                            with st.container():
                                                # Grading Report
                                                st.markdown("### 📝 Grading Report")
                                                st.markdown(st.session_state[report_key], unsafe_allow_html=False)
    
                                                # Instructions and Action Buttons
                                                st.markdown("### 📝 Grade Submission")
                                                st.markdown("""
                                                **Instructions:**
                                                1. Review the grading report above.
                                                2. Click the button below to open Google Classroom.
                                                3. Enter the grade based on the report.
                                                4. Click 'Return' in Google Classroom to submit the grade.
                                                """)
    
                                                # Action Buttons
                                                action_col1, action_col2 = st.columns([1, 3])
                                                if action_col1.button("🔗 Open in Google Classroom", key=f"classroom_link_{submission_id}"):
                                                    classroom_url = submission.get('alternate_link', '#')
                                                    webbrowser.open(classroom_url)
    
                                                if action_col2.button("❌ Cancel Grading", key=f"cancel_{submission_id}"):
                                                    st.session_state[grading_key] = False
                                                    st.session_state[report_key] = None
                                                    st.rerun()
    
                                    except Exception as e:
                                        st.error(f"Error during grading: {str(e)}")
                                        st.session_state[grading_key] = False
                                        st.session_state[report_key] = None
            else:
                st.write("No assignments found.")
    
            # Announcements Section
            st.subheader("📢 Announcements")
            if course.get('messages'):
                for message in course['messages']:
                    st.write(f"- {message}")
            else:
                st.write("No announcements found.")

# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="Google Classroom Dashboard", layout="wide")
    st.title("📚 Google Classroom Dashboard with Chatbot 🤖")

    # Initialize session state variables
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'authorized' not in st.session_state:
        st.session_state.authorized = False
    if 'courses_data' not in st.session_state:
        st.session_state.courses_data = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'chatbot_ready' not in st.session_state:
        st.session_state.chatbot_ready = False
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'timestamp' not in st.session_state:
        st.session_state.timestamp = None
    if 'service' not in st.session_state:
        st.session_state.service = None
    if 'drive_service' not in st.session_state:
        st.session_state.drive_service = None
    if 'show_role_selection' not in st.session_state:
        st.session_state.show_role_selection = True

    # Create custom CSS for buttons
    st.markdown("""
        <style>
        div.stButton > button {
            width: 100%;
            height: 100px;
            font-size: 50px;
            font-weight: bold;
            margin: 10px 0px;
            border-radius: 10px;
        }
        div.stButton > button:hover {
            transform: scale(1.02);
            transition: all 0.1s ease-in-out;
        }
        .change-role-btn {
            margin: 10px 0;
            padding: 5px 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar setup
    st.sidebar.header("Dashboard Controls")

    # Show role selection if not authenticated or if explicitly showing role selection
    if not st.session_state.authorized or st.session_state.show_role_selection:
        st.sidebar.subheader("Select Your Role")
        
        # Role selection buttons in two columns
        col1, col2 = st.sidebar.columns(2)
        
        # Role selection buttons with visual feedback
        if col1.button("👨‍🎓\nStudent", 
                       type="primary" if st.session_state.role == "Student" else "secondary",
                       use_container_width=True):
            st.session_state.role = "Student"
            st.session_state.show_role_selection = False
            # Reset auth state when changing roles
            st.session_state.authorized = False
            st.session_state.courses_data = []
            st.session_state.chatbot_ready = False

        if col2.button("👩‍🏫\nTeacher", 
                       type="primary" if st.session_state.role == "Teacher" else "secondary",
                       use_container_width=True):
            st.session_state.role = "Teacher"
            st.session_state.show_role_selection = False
            # Reset auth state when changing roles
            st.session_state.authorized = False
            st.session_state.courses_data = []
            st.session_state.chatbot_ready = False
    
    # Show current role and change role button if authenticated
    if st.session_state.authorized and st.session_state.role:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current Role")
        st.sidebar.write(f"You are logged in as: **{st.session_state.role}**")
        if st.sidebar.button("🔄 Change Role", key="change_role"):
            # Reset all necessary session state variables
            st.session_state.show_role_selection = True
            st.session_state.role = None
            st.session_state.authorized = False
            st.session_state.courses_data = []
            st.session_state.chatbot_ready = False
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None

    # Only show authentication after role selection and if not already authenticated
    if st.session_state.role and not st.session_state.authorized and not st.session_state.show_role_selection:
        st.sidebar.markdown("---")
        st.sidebar.header("Authentication")
        if st.sidebar.button("🔑 Authenticate with Google Classroom"):
            # Start Flask server in a separate thread
            flask_thread = threading.Thread(target=run_flask_app, daemon=True)
            flask_thread.start()

            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRET_FILE, SCOPES, redirect_uri=REDIRECT_URI
                )
            except FileNotFoundError:
                st.error(f"Client secret file '{CLIENT_SECRET_FILE}' not found.")
                return

            auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
            webbrowser.open(auth_url)
            st.info("Opening browser for authentication...")

            # Wait for authorization
            with st.spinner("Waiting for authorization..."):
                authorization_complete.wait(timeout=300)  # Wait up to 5 minutes

            if auth_code:
                try:
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                    st.session_state.creds = creds
                    st.session_state.authorized = True
                    st.success("✅ Authentication successful!")
                except Exception as e:
                    st.error(f"Error fetching token: {e}")
            else:
                st.error("❌ Authorization failed or timed out.")

    # If authorized, fetch data and initialize services
    if st.session_state.authorized and not st.session_state.courses_data:
        try:
            # Initialize Classroom API service
            st.session_state.service = build('classroom', 'v1', credentials=st.session_state.creds)
            service = st.session_state.service

            # Initialize Drive API service
            st.session_state.drive_service = build('drive', 'v3', credentials=st.session_state.creds)
            drive_service = st.session_state.drive_service

            with st.spinner("Fetching courses from Google Classroom..."):
                courses_data = fetch_courses_data(service)
            st.session_state.courses_data = courses_data
            st.success("✅ Courses data fetched successfully!")

            # Create Document objects
            documents = create_classroom_documents(st.session_state.courses_data)
            st.session_state.documents = documents
            st.session_state.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.success("📄 Data ingested successfully!")

            # Setup Chatbot only for Students
            if st.session_state.role == "Student":
                with st.spinner("Setting up chatbot... This may take a few minutes."):
                    vectorstore, qa_chain = setup_chatbot(st.session_state.documents, st.session_state.timestamp)
                    if vectorstore and qa_chain:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.qa_chain = qa_chain
                        st.session_state.chatbot_ready = True
                        st.success("🤖 Chatbot is ready!")
            else:
                st.session_state.chatbot_ready = False  # Ensure chatbot is not ready for teachers
        except Exception as e:
            st.error(f"Error during data ingestion or chatbot setup: {e}")

    # Display based on role
    if st.session_state.authorized:
        if st.session_state.role == "Student":
            display_student_interface(st)
        elif st.session_state.role == "Teacher":
            service = st.session_state.service
            drive_service = st.session_state.drive_service
            display_teacher_interface(st, service, drive_service)

    st.markdown("---")
    st.markdown("Developed by SG")


# -------------------------------
# Run the App
# -------------------------------
if __name__ == '__main__':
    main()
