import streamlit as st
import google.generativeai as genai
import os
import tempfile
import time
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import logging
import re

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration ---
st.set_page_config(
    page_title="AI League Scout Eye (Gemini Flex - عربي)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants (Arabic) ---
# Skill names (used for prompts and internal logic - keep English keys)
SKILLS_TO_EVALUATE_EN = [
    "Jumping", "Running", "Passing", "Receiving", "Zigzag"
]
# Skill labels (Arabic - for display in UI)
SKILLS_LABELS_AR = {
    "Jumping": "القفز بالكرة (تنطيط الركبة)",
    "Running": "الجري بالكرة (التحكم)",
    "Passing": "التمرير",
    "Receiving": "استقبال الكرة",
    "Zigzag": "المراوغة (زجزاج)"
}
MAX_SCORE_PER_SKILL = 5
MODEL_NAME = "models/gemini-1.5-pro"

# Analysis Modes (Arabic)
MODE_MULTI_VIDEO_AR = "تقييم جميع المهارات الخمس (5 فيديوهات منفصلة)"
MODE_SINGLE_VIDEO_ALL_SKILLS_AR = "تقييم جميع المهارات الخمس (فيديو واحد)"
MODE_SINGLE_VIDEO_ONE_SKILL_AR = "تقييم مهارة محددة (فيديو واحد)"

# --- Gemini API Configuration ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    
    genai.configure(api_key=api_key)
    # st.success("🔑 تم تحميل مفتاح Gemini API بنجاح.") # Optional success message
    logging.info("Gemini API Key loaded successfully.")
except KeyError:
    st.error("❗️ لم يتم العثور على مفتاح Gemini API في أسرار Streamlit. الرجاء إضافة `GEMINI_API_KEY`.")
    st.stop()
except Exception as e:
    st.error(f"❗️ فشل في إعداد Gemini API: {e}")
    logging.error(f"Gemini API configuration failed: {e}")
    st.stop()

# --- Gemini Model Setup ---
@st.cache_resource
def load_gemini_model():
    try:
        generation_config = {
            "temperature": 0.1, "top_p": 1, "top_k": 1,
            "max_output_tokens": 50,
        }

        # --- WARNING: LOWEST SAFETY SETTINGS ---
        # Use BLOCK_NONE cautiously, only for testing safe content.
        # This minimizes blocking but increases risk of harmful content.
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        # --- End of Warning Section ---

        model = genai.GenerativeModel(
            model_name=MODEL_NAME, # Ensure MODEL_NAME is set correctly (e.g., "gemini-1.5-pro-latest")
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logging.info(f"Gemini Model '{MODEL_NAME}' loaded with MINIMUM safety settings (BLOCK_NONE).")
        # st.success(f"✅ تم تحميل نموذج Gemini '{MODEL_NAME}' بأقل إعدادات أمان.") # Optional
        return model
    except Exception as e:
        st.error(f"❗️ فشل تحميل نموذج Gemini '{MODEL_NAME}': {e}")
        logging.error(f"Gemini model loading failed: {e}")
        return None

model = load_gemini_model()
if not model:
    st.stop()

# --- CSS Styling (Remains the same) ---
st.markdown("""
<style>
    /* ... (CSS styles remain the same) ... */
    body { direction: rtl; } /* Add Right-to-Left direction */
    .stApp { background-color: #1a1a4a; color: white; }
    .stButton>button { background-color: transparent; color: white; border: 1px solid transparent; padding: 15px 25px; text-align: center; text-decoration: none; display: inline-block; font-size: 1.2em; margin: 15px 10px; cursor: pointer; transition: background-color 0.3s ease, border-color 0.3s ease; font-weight: bold; border-radius: 8px; min-width: 200px; font-family: 'Tajawal', sans-serif; /* Example Arabic font */ }
    .stButton>button:hover { background-color: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.4); }
    .stButton>button:active { background-color: rgba(255, 255, 255, 0.2); }
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; display: flex; flex-direction: column; align-items: center; }
    .title-text { font-size: 3em; font-weight: bold; color: white; text-align: center; margin-bottom: 0.3em; }
    .slogan-text { font-size: 1.8em; font-weight: bold; color: white; text-align: center; margin-bottom: 1.5em; } /* Direction set globally */
     h2 { color: #d8b8d8; text-align: center; margin-top: 1.5em; font-size: 2em; font-weight: bold; }
     h3 { color: white; text-align: center; margin-top: 1em; font-size: 1.5em; }
    /* Adjust file uploader label alignment for RTL */
    .stFileUploader label { color: white !important; font-size: 1.1em !important; font-weight: bold; text-align: right !important; width: 100%; padding-left: 10px; /* Add padding if needed */ }
    .stMetric { background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; text-align: center; }
     .stMetric label { font-weight: bold; color: #d8b8d8; }
     .stMetric div[data-testid="metric-value"] { font-size: 2em; font-weight: bold; color: white; }
    img[alt="matplotlib chart"] { background-color: transparent !important; }
    /* Style radio buttons for RTL */
    div[role="radiogroup"] { display: flex; flex-direction: row-reverse; justify-content: center; margin-bottom: 1em;}
    div[role="radiogroup"] label { color: white; font-size: 1.1em; margin-left: 15px; margin-right: 5px; /* Adjust spacing */ }
    /* Center selectbox */
    .stSelectbox div[data-baseweb="select"] > div { background-color: rgba(255, 255, 255, 0.1); border-radius: 8px; color: white; border: 1px solid rgba(255, 255, 255, 0.3); }
    .stSelectbox label { color: white !important; font-weight: bold; text-align: center !important; width: 100%; margin-bottom: 5px;}

</style>
""", unsafe_allow_html=True)

# =========== Gemini Interaction Functions ============================

#def create_prompt_for_skill(skill_key_en):
#    """
#    Generates a specific prompt for Gemini based on the ENGLISH skill key.
#    (Simpler version without detailed rubrics, using 'base_instruction').
#    """
#    skill_name_ar = SKILLS_LABELS_AR.get(skill_key_en, skill_key_en) # Get Arabic name for prompt clarity
#
#    # Define the core instructions in the base_instruction variable
#    base_instruction = f"""
#    حلل مقطع الفيديو المقدم مع التركيز فقط على مهارة كرة القدم: {skill_name_ar}.
#    قم بتقييم أداء اللاعب بناءً على المعايير المعتادة لهذه المهارة (التحكم، التنفيذ، الفعالية).
#    عين درجة رقمية من 0 إلى {MAX_SCORE_PER_SKILL}، حيث 0 تعني ضعيف جدًا/لم يحاول و {MAX_SCORE_PER_SKILL} تعني تنفيذ ممتاز.
#
#    هام جدًا: قم بالرد بالدرجة الرقمية الصحيحة فقط (مثال: "3" أو "5"). لا تقم بتضمين أي شروحات أو أوصاف أو أي نص آخر. فقط الرقم.
#    """
#
#    # In this simple version, we just return the base instruction directly
#    # No extra hints or rubrics are added.
#    return base_instruction
def create_prompt_for_skill(skill_key_en):
    """Generates a specific prompt WITH a detailed rubric for Gemini."""
    skill_name_ar = SKILLS_LABELS_AR.get(skill_key_en, skill_key_en)

    # --- Define Rubrics (Translate criteria carefully) ---
    rubrics = {
        "Jumping": """
        **معايير تقييم القفز بالكرة (تنطيط الركبة):**
        - 0: لا توجد محاولات أو لمسات ناجحة بالركبة أثناء الطيران.
        - 1: لمسة واحدة ناجحة بالركبة أثناء الطيران، مع تحكم ضعيف.
        - 2: لمستان ناجحتان بالركبة أثناء الطيران، تحكم مقبول.
        - 3: ثلاث لمسات ناجحة بالركبة، تحكم جيد وثبات.
        - 4: أربع لمسات ناجحة، تحكم ممتاز وثبات هوائي جيد.
        - 5: خمس لمسات أو أكثر، تحكم استثنائي، إيقاع وثبات ممتازين.
        """,
        "Running": """
        **معايير تقييم الجري بالكرة:**
        - 0: تحكم ضعيف جدًا، الكرة تبتعد كثيرًا عن القدم.
        - 1: تحكم ضعيف، الكرة تبتعد بشكل ملحوظ أحيانًا.
        - 2: تحكم مقبول، الكرة تبقى ضمن نطاق واسع حول اللاعب.
        - 3: تحكم جيد، الكرة تبقى قريبة بشكل عام أثناء الجري.
        - 4: تحكم جيد جدًا، الكرة قريبة باستمرار حتى مع تغيير السرعة.
        - 5: تحكم ممتاز، الكرة تبدو ملتصقة بالقدم، سيطرة كاملة.
        """,
        "Passing": """
        **معايير تقييم التمرير:**
        - 0: تمريرة خاطئة تمامًا أو ضعيفة جدًا أو بدون دقة.
        - 1: تمريرة بدقة ضعيفة أو قوة غير مناسبة بشكل كبير.
        - 2: تمريرة مقبولة تصل للهدف ولكن بقوة أو دقة متوسطة.
        - 3: تمريرة جيدة ودقيقة بقوة مناسبة للمسافة.
        - 4: تمريرة دقيقة جدًا ومتقنة بقوة مثالية.
        - 5: تمريرة استثنائية، دقة وقوة مثالية، تضع المستلم في موقف جيد.
        """,
        "Receiving": """
        **معايير تقييم استقبال الكرة:**
        - 0: فشل في السيطرة على الكرة تمامًا عند الاستقبال.
        - 1: لمسة أولى سيئة، الكرة تبتعد كثيرًا أو تتطلب جهدًا للسيطرة عليها.
        - 2: استقبال مقبول، الكرة تحت السيطرة بعد لمستين أو بحركة إضافية.
        - 3: استقبال جيد، لمسة أولى نظيفة تبقي الكرة قريبة.
        - 4: استقبال جيد جدًا، لمسة أولى ممتازة تهيئ الكرة للخطوة التالية بسهولة.
        - 5: استقبال استثنائي، لمسة أولى مثالية، تحكم فوري وسلس.
        """,
        "Zigzag": """
        **معايير تقييم المراوغة (زجزاج):**
        - 0: فقدان السيطرة على الكرة عند محاولة تغيير الاتجاه.
        - 1: تغيير اتجاه بطيء مع ابتعاد الكرة عن القدم.
        - 2: تغيير اتجاه مقبول مع الحفاظ على الكرة ضمن نطاق تحكم واسع.
        - 3: تغيير اتجاه جيد مع إبقاء الكرة قريبة نسبيًا.
        - 4: تغيير اتجاه سريع مع إبقاء الكرة قريبة جدًا من القدم.
        - 5: تغيير اتجاه خاطف وسلس مع سيطرة تامة على الكرة (تبدو ملتصقة بالقدم).
        """
    }

    specific_rubric = rubrics.get(skill_key_en, "لا توجد معايير محددة لهذه المهارة.")

    # --- Construct the Final Prompt ---
    prompt = f"""
    مهمتك هي تقييم مهارة كرة القدم '{skill_name_ar}' المعروضة في الفيديو.
    استخدم المعايير التالية لتقييم الأداء وتحديد درجة من 0 إلى {MAX_SCORE_PER_SKILL}:

    {specific_rubric}

    شاهد الفيديو بعناية. بناءً على المعايير المذكورة أعلاه فقط، ما هي الدرجة التي تصف أداء اللاعب بشكل أفضل؟

    هام جدًا: قم بالرد بالدرجة الرقمية الصحيحة فقط (مثال: "3" أو "5"). لا تقم بتضمين أي شروحات أو أوصاف أو أي نص آخر. فقط الرقم.
    """
    return prompt

def upload_and_wait_gemini(video_path, display_name="video_upload", status_placeholder=st.empty()):
    """Uploads video, waits for ACTIVE state, returns file object or None."""
    uploaded_file = None
    status_placeholder.info(f"⏳ جاري رفع الفيديو '{os.path.basename(video_path)}'...")
    logging.info(f"Starting upload for {display_name}")
    try:
        # Use a temporary display name if the original causes issues
        safe_display_name = f"upload_{int(time.time())}"
        uploaded_file = genai.upload_file(path=video_path, display_name=safe_display_name) # Use safe name
        status_placeholder.info(f"📤 اكتمل الرفع لـ '{display_name}'. برجاء الانتظار للمعالجة...")
        logging.info(f"Upload API call successful for {display_name}, file name: {uploaded_file.name}. Waiting for ACTIVE state.")

        timeout = 240
        start_time = time.time()
        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                logging.error(f"Timeout waiting for file processing for {uploaded_file.name}")
                raise TimeoutError("انتهت مهلة معالجة الفيديو.")
            time.sleep(10)
            uploaded_file = genai.get_file(uploaded_file.name)
            logging.debug(f"File {uploaded_file.name} state: {uploaded_file.state.name}")

        if uploaded_file.state.name == "FAILED":
            logging.error(f"File processing failed for {uploaded_file.name}")
            raise ValueError("فشلت معالجة الفيديو من جانب Google.")
        elif uploaded_file.state.name != "ACTIVE":
             logging.error(f"Unexpected file state {uploaded_file.state.name} for {uploaded_file.name}")
             raise ValueError(f"حالة ملف فيديو غير متوقعة: {uploaded_file.state.name}")

        status_placeholder.success(f"✅ الفيديو '{display_name}' جاهز للتحليل.")
        logging.info(f"File {uploaded_file.name} is ACTIVE.")
        return uploaded_file

    except Exception as e:
        status_placeholder.error(f"❌ خطأ أثناء رفع/معالجة الفيديو لـ '{display_name}': {e}")
        logging.error(f"Upload/Wait failed for '{display_name}': {e}", exc_info=True)
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
                logging.info(f"Cleaned up partially processed/failed file: {uploaded_file.name}")
            except Exception as del_e:
                 logging.warning(f"Failed to delete file {uploaded_file.name} after upload error: {del_e}")
        return None

def analyze_video_with_prompt(gemini_file_obj, skill_key_en, status_placeholder=st.empty()):
    """
    Analyzes an ACTIVE video file object with a specific skill prompt,
    includes debugging output, and handles potential empty/blocked responses.
    """
    score = 0 # Default score
    skill_name_ar = SKILLS_LABELS_AR.get(skill_key_en, skill_key_en) # For status message
    prompt = create_prompt_for_skill(skill_key_en) # Assumes create_prompt_for_skill is defined elsewhere

    # Initial status message before API call
    status_placeholder.info(f"🧠 جاري تحليل مهارة '{skill_name_ar}' بواسطة Gemini...")
    logging.info(f"Requesting analysis for skill '{skill_key_en}' using file {gemini_file_obj.name}")

    try:
        # --- Make the API call ---
        response = model.generate_content([prompt, gemini_file_obj], request_options={"timeout": 120})

        # --- START DEBUG OUTPUT ---
        # Log the full response object regardless (for detailed inspection in logs)
        logging.info(f"Full Gemini Response Object for {skill_key_en}: {response}")

        # Try to display debug info in the UI
        try:
            # Use a container within the status placeholder to show debug info temporarily
            with status_placeholder.container():
                 st.divider() # Visual separator
                 st.error(f"--- DEBUG INFO: Response for '{skill_name_ar}' ---") # Make it stand out
                 st.write("**Prompt Feedback (Safety Check):**")
                 st.json(response.prompt_feedback.__str__()) # Try to display as string/json if possible
                 # st.write(response.prompt_feedback) # Alternative display
                 st.write("**Candidates List (Generated Content):**")
                 st.json(response.candidates.__str__()) # Try to display as string/json
                 # st.write(response.candidates) # Alternative display
                 st.error("--- END DEBUG INFO ---")
                 st.divider() # Visual separator
        except Exception as debug_e:
            # If displaying debug info itself fails, log it and show a simple UI message
            logging.error(f"Error displaying debug info for {skill_key_en}: {debug_e}", exc_info=True)
            with status_placeholder.container():
                st.warning(f"⚠️ تعذر عرض معلومات التصحيح التفصيلية لـ '{skill_name_ar}'.")
                st.divider()

        # Re-display the main status message after attempting debug output
        status_placeholder.info(f"🧠 جاري تحليل مهارة '{skill_name_ar}' بواسطة Gemini...")
        # --- END DEBUG OUTPUT ---


        # --- START RESPONSE CHECKING & PARSING ---
        prompt_blocked = False
        block_reason_text = "N/A"
        try:
            # Check block_reason safely - this attribute exists if the prompt itself was blocked
            if response.prompt_feedback.block_reason:
                prompt_blocked = True
                block_reason_text = str(response.prompt_feedback.block_reason)
                # Display block reason in UI (no need to log again, already in full response log)
                st.warning(f"⚠️ استجابة Gemini محظورة لـ '{skill_name_ar}' بسبب: {block_reason_text}. النتيجة=0.")
                # Log just the fact it was blocked by prompt feedback for quick filtering
                logging.warning(f"Response blocked (prompt feedback) for {skill_key_en}: {block_reason_text}")
        except AttributeError:
            logging.debug(f"No block_reason found in prompt_feedback for {skill_key_en}.") # Normal if not blocked
        except Exception as feedback_err:
             st.warning(f"⚠️ خطأ غير متوقع في الوصول إلى ملاحظات الطلب (prompt_feedback) لـ '{skill_name_ar}': {feedback_err}.")
             logging.warning(f"Unexpected error accessing prompt_feedback for {skill_key_en}: {feedback_err}.")

        # Check if candidates list is empty OR if the prompt was blocked earlier
        # Note: response.candidates might be empty *even if* prompt_blocked is False, if generation failed/was blocked by safety AFTER prompt phase
        if prompt_blocked or not response.candidates:
             # Only show the "empty candidates" warning if it wasn't already flagged as blocked by prompt
             if not response.candidates and not prompt_blocked:
                 st.warning(f"⚠️ استجابة Gemini فارغة (لا توجد مرشحات) لـ '{skill_name_ar}'. النتيجة=0.")
                 logging.warning(f"Response candidates list is empty for {skill_key_en} (and prompt was not blocked).")
             # Score is 0 if blocked OR if candidates is empty
             score = 0

        # If NOT blocked AND candidates exist, attempt to parse the content
        else:
            try:
                # Access the text content safely now
                raw_score_text = response.text.strip()
                logging.info(f"Gemini Raw Response Text for {skill_key_en}: '{raw_score_text}'")

                # Use regex to find the first sequence of digits
                match = re.search(r'\d+', raw_score_text)
                if match:
                    score = int(match.group(0))
                    score = max(0, min(MAX_SCORE_PER_SKILL, score)) # Clamp score
                    # Update status placeholder to success ONLY if score is parsed
                    status_placeholder.success(f"✅ اكتمل تحليل '{skill_name_ar}'. النص الخام: '{raw_score_text}', النتيجة: {score}")
                    logging.info(f"Analysis for {skill_key_en} successful. Raw: '{raw_score_text}', Score: {score}")
                else:
                     # Text received, but no number found
                     st.warning(f"⚠️ لم يتم العثور على رقم في استجابة Gemini لـ '{skill_name_ar}' ('{raw_score_text}'). النتيجة=0.")
                     logging.warning(f"Could not parse score for {skill_key_en} from text: '{raw_score_text}' - no digits found.")
                     score = 0

            except ValueError as e_parse: # Error accessing .text or during int() conversion
                 candidate_text_fallback = "Error retrieving text"
                 try: # Safely attempt to get fallback text for logging
                     if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                         candidate_text_fallback = response.candidates[0].content.parts[0].text
                 except Exception: pass

                 st.warning(f"⚠️ لم نتمكن من تحليل النتيجة من استجابة Gemini لـ '{skill_name_ar}'. الخطأ: {e_parse}. النص التقريبي: '{candidate_text_fallback}'. النتيجة=0.")
                 logging.warning(f"Could not parse score for {skill_key_en}. Access/Parse Error: {e_parse}. Fallback Text: '{candidate_text_fallback}'. Candidates: {response.candidates}")
                 score = 0
            except Exception as e_generic_parse: # Catch other unexpected parsing errors
                 st.error(f"⚠️ خطأ غير متوقع أثناء تحليل استجابة Gemini لـ '{skill_name_ar}': {e_generic_parse}. النتيجة=0.")
                 logging.error(f"Unexpected parsing error for {skill_key_en}: {e_generic_parse}", exc_info=True)
                 score = 0
        # --- END RESPONSE CHECKING & PARSING ---

    # --- Outer Exception Handling for API Call/Generation Issues ---
    except genai.types.generation_types.BlockedPromptException as bpe:
         st.error(f"❌ تم حظر طلب Gemini API نفسه لـ '{skill_name_ar}': {bpe}")
         logging.error(f"Prompt blocked during API call for {skill_key_en}: {bpe}")
         score = 0
    except genai.types.generation_types.StopCandidateException as sce:
         st.error(f"❌ توقف إنشاء الاستجابة بشكل غير متوقع لـ '{skill_name_ar}' (أمان/سياسة): {sce}")
         logging.error(f"Analysis stopped (safety/policy) for {skill_key_en}: {sce}")
         score = 0
    except TimeoutError as te: # This assumes the timeout error from the API call itself
         st.error(f"❌ انتهت مهلة استدعاء Gemini API لـ '{skill_name_ar}': {te}")
         logging.error(f"Timeout during API call/generation for {skill_key_en}: {te}")
         score = 0
    except Exception as e:
        # Catch-all for other potential errors (e.g., network issues, API config)
        st.error(f"❌ حدث خطأ عام أثناء استدعاء Gemini API لـ '{skill_name_ar}': {e}")
        logging.error(f"General Gemini analysis failed for {skill_key_en}: {e}", exc_info=True)
        score = 0 # Ensure score is 0 on any failure

    # Return the determined score (0 if any error occurred or parsing failed)
    return score

def delete_gemini_file(gemini_file_obj, status_placeholder=st.empty()):
    """Deletes the uploaded file from Gemini Cloud Storage."""
    if not gemini_file_obj: return
    try:
        display_name = gemini_file_obj.display_name or gemini_file_obj.name
        status_placeholder.info(f"🗑️ جاري حذف الملف المرفوع '{display_name}' من التخزين السحابي...")
        logging.info(f"Attempting to delete cloud file: {gemini_file_obj.name}")
        genai.delete_file(gemini_file_obj.name)
        status_placeholder.empty() # Clear the message after deletion
        logging.info(f"Cloud file deleted successfully: {gemini_file_obj.name}")
    except Exception as e:
        st.warning(f"⚠️ لم نتمكن من حذف الملف السحابي {gemini_file_obj.name}: {e}")
        logging.warning(f"Could not delete cloud file {gemini_file_obj.name}: {e}")


# =========== Grading and Plotting Functions (Arabic labels adjusted) =================

def evaluate_final_grade_from_individual_scores(scores_dict):
    # ... (Grading logic remains the same) ...
    total = sum(scores_dict.values())
    max_possible = len(scores_dict) * MAX_SCORE_PER_SKILL
    percentage = (total / max_possible) * 100 if max_possible > 0 else 0
    if percentage >= 90: grade = 'ممتاز (A)'
    elif percentage >= 75: grade = 'جيد جداً (B)'
    elif percentage >= 55: grade = 'جيد (C)'
    elif percentage >= 40: grade = 'مقبول (D)'
    else: grade = 'ضعيف (F)'
    return {"scores": scores_dict, "total_score": total, "grade": grade, "max_score": max_possible}

def plot_results(results):
    """Generates and returns a matplotlib figure of the scores with Arabic labels."""
    # Use the Arabic labels directly from SKILLS_LABELS_AR based on the keys in results['scores']
    try:
        reshaped_labels = {}
        for key_en in results['scores'].keys():
            label_ar = SKILLS_LABELS_AR.get(key_en, key_en) # Get Arabic label
            reshaped_labels[key_en] = get_display(arabic_reshaper.reshape(label_ar))

        labels_for_plot = [reshaped_labels[key_en] for key_en in results['scores'].keys()] # Ensure order matches scores
        plot_title = get_display(arabic_reshaper.reshape(f"التقييم النهائي - التقدير: {results['grade']} ({results['total_score']}/{results['max_score']})"))
        y_axis_label = get_display(arabic_reshaper.reshape(f"الدرجة (من {MAX_SCORE_PER_SKILL})"))
    except Exception as e:
        st.warning(f"Could not reshape Arabic text for plot: {e}")
        logging.warning(f"Arabic reshaping failed: {e}")
        labels_for_plot = list(results['scores'].keys()) # Fallback to English keys
        plot_title = f"Final Evaluation - Grade: {results['grade']} ({results['total_score']}/{results['max_score']})"
        y_axis_label = f"Score (out of {MAX_SCORE_PER_SKILL})"

    scores = list(results['scores'].values())
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels_for_plot, scores)
    ax.set_ylim(0, MAX_SCORE_PER_SKILL + 0.5)
    ax.set_ylabel(y_axis_label, fontsize=12, fontweight='bold')
    ax.set_title(plot_title, fontsize=14, fontweight='bold')

    colors = ['green' if s >= 4 else 'orange' if s >= 2.5 else 'red' for s in scores]
    for bar, color in zip(bars, colors): bar.set_color(color)
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval}', ha='center', va='bottom', fontsize=11)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=11, rotation=15, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))
    plt.tight_layout()
    return fig

# =========== Streamlit App Layout (Arabic) ====================================

# Initialize session state
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = None
if 'analysis_mode' not in st.session_state: st.session_state.analysis_mode = MODE_MULTI_VIDEO_AR # Default mode
if 'selected_skill' not in st.session_state: st.session_state.selected_skill = SKILLS_TO_EVALUATE_EN[0] # Default skill (English key)
if 'uploaded_files_state' not in st.session_state: st.session_state.uploaded_files_state = {} # Store file objects

# --- Top Row: Logo ---
col1, col_mid, col2 = st.columns([1, 3, 1])
with col1:
    st.markdown("<p style='font-size: 1.2em; font-weight: bold;'>AI LEAGUE</p>", unsafe_allow_html=True)

# --- Center Area: Main Logo, Title, Slogan ---
st.container()
st.markdown("<h1 style='text-align: center; color: white; margin-top: 20px;'>Scout Eye</h1>", unsafe_allow_html=True)
st.markdown("<div class='title-text'>عين الكشاف</div>", unsafe_allow_html=True)
st.markdown("<div class='slogan-text'>نكتشف ، نحمي ، ندعم</div>", unsafe_allow_html=True)

# --- Bottom Area: Clickable Options (Arabic) ---
st.container()
col_b1, col_b2, col_b3 = st.columns(3)
button_keys = ["btn_person", "btn_star", "btn_legend"]

if col_b1.button("✔️ الشخص المناسب", key=button_keys[0]):
    st.session_state.page = 'الشخص_المناسب'
    st.session_state.evaluation_results = None
if col_b2.button("⭐ نجم لا يغيب", key=button_keys[1]):
    st.session_state.page = 'نجم_لا_يغيب'
    st.session_state.evaluation_results = None
if col_b3.button("⚽ إسطورة الغد", key=button_keys[2]):
    st.session_state.page = 'اسطورة_الغد'
    st.session_state.evaluation_results = None

# --- Conditional Page Content ---

# ==================================
# ==      إسطورة الغد Page       ==
# ==================================
if st.session_state.page == 'اسطورة_الغد':
    st.markdown("---")
    st.markdown("## ⚽ إسطورة الغد - تحليل المهارات بواسطة Gemini ⚽")

    # --- Mode Selection ---
    st.session_state.analysis_mode = st.radio(
        "اختر طريقة التحليل:",
        options=[MODE_MULTI_VIDEO_AR, MODE_SINGLE_VIDEO_ALL_SKILLS_AR, MODE_SINGLE_VIDEO_ONE_SKILL_AR],
        index=[MODE_MULTI_VIDEO_AR, MODE_SINGLE_VIDEO_ALL_SKILLS_AR, MODE_SINGLE_VIDEO_ONE_SKILL_AR].index(st.session_state.analysis_mode), # Persist selection
        key="analysis_mode_radio"
    )

    st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.3);'>", unsafe_allow_html=True)

    # --- UI based on Mode ---
    uploaded_file_objects = {} # Dict to hold current file objects
    skill_to_analyze_key_en = None # English key for the selected skill

    if st.session_state.analysis_mode == MODE_MULTI_VIDEO_AR:
        st.markdown("<p style='text-align: center; font-size: 1.1em;'>الرجاء رفع فيديو منفصل لكل مهارة من المهارات الخمس:</p>", unsafe_allow_html=True)
        col_upload1, col_upload2 = st.columns(2)
        with col_upload1:
            for skill_key in SKILLS_TO_EVALUATE_EN[:3]:
                label = f"{list(SKILLS_TO_EVALUATE_EN).index(skill_key)+1}. {SKILLS_LABELS_AR[skill_key]}"
                uploaded_file_objects[skill_key] = st.file_uploader(label, type=["mp4", "avi", "mov", "mkv", "webm"], key=f"upload_multi_{skill_key}")
        with col_upload2:
            for skill_key in SKILLS_TO_EVALUATE_EN[3:]:
                label = f"{list(SKILLS_TO_EVALUATE_EN).index(skill_key)+1}. {SKILLS_LABELS_AR[skill_key]}"
                uploaded_file_objects[skill_key] = st.file_uploader(label, type=["mp4", "avi", "mov", "mkv", "webm"], key=f"upload_multi_{skill_key}")
        # Check if all 5 are uploaded for this mode
        ready_to_analyze = all(uploaded_file_objects.get(skill) for skill in SKILLS_TO_EVALUATE_EN)

    elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
        st.markdown("<p style='text-align: center; font-size: 1.1em;'>الرجاء رفع فيديو واحد لتقييم جميع المهارات الخمس:</p>", unsafe_allow_html=True)
        single_video_all = st.file_uploader("📂 ارفع الفيديو الشامل:", type=["mp4", "avi", "mov", "mkv", "webm"], key="upload_single_all")
        if single_video_all:
            for skill_key in SKILLS_TO_EVALUATE_EN:
                uploaded_file_objects[skill_key] = single_video_all # Assign same file object to all skills
        ready_to_analyze = single_video_all is not None

    elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ONE_SKILL_AR:
        st.markdown("<p style='text-align: center; font-size: 1.1em;'>الرجاء رفع فيديو واختيار المهارة المراد تقييمها:</p>", unsafe_allow_html=True)
        col_select, col_upload = st.columns([1, 2])
        with col_select:
            # Use Arabic labels for selection, but store the English key
            skill_to_analyze_key_en = st.selectbox(
                "اختر المهارة:",
                options=SKILLS_TO_EVALUATE_EN,
                format_func=lambda key: SKILLS_LABELS_AR.get(key, key), # Show Arabic label
                key="select_single_skill"
            )
        with col_upload:
            single_video_one = st.file_uploader(f"📂 ارفع فيديو مهارة '{SKILLS_LABELS_AR.get(skill_to_analyze_key_en, '')}'", type=["mp4", "avi", "mov", "mkv", "webm"], key="upload_single_one")

        if single_video_one and skill_to_analyze_key_en:
             uploaded_file_objects[skill_to_analyze_key_en] = single_video_one
        ready_to_analyze = single_video_one is not None and skill_to_analyze_key_en is not None

    # Store current state of uploaded files for potential re-runs without re-uploading if desired
    st.session_state.uploaded_files_state = uploaded_file_objects

    st.markdown("---")

    # --- Analysis Button ---
    if st.button("🚀 بدء التحليل بواسطة Gemini", key="start_gemini_eval", disabled=not ready_to_analyze):
        try:
            st.info("⏳ جاري التهيئة لبدء تحليل Gemini...")
            results_dict = {}
            error_occurred = False
            local_temp_files = {} # Paths of local temp video files
            gemini_file_objects = {} # Store ACTIVE Gemini file objects {skill_key: file_obj}
            status_global = st.empty() # For global status messages

            # --- Step 1: Upload and Process Videos ---
            upload_successful = True
            with st.status("📤 رفع ومعالجة الفيديوهات...", expanded=True) as status_upload:
                try:
                    # Handle single video upload first if applicable
                    if st.session_state.analysis_mode in [MODE_SINGLE_VIDEO_ALL_SKILLS_AR, MODE_SINGLE_VIDEO_ONE_SKILL_AR]:
                        skill_key_to_process = list(st.session_state.uploaded_files_state.keys())[0] # Get the relevant skill key
                        uploaded_file_obj = st.session_state.uploaded_files_state[skill_key_to_process]
                        if uploaded_file_obj:
                            status_upload.write(f"رفع ومعالجة الفيديو الواحد...")
                            # Save to temp local file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1]) as tmp_file:
                                tmp_file.write(uploaded_file_obj.getvalue())
                                video_path = tmp_file.name
                                local_temp_files['single'] = video_path # Store path

                            # Upload to Gemini and wait
                            gemini_file = upload_and_wait_gemini(video_path, uploaded_file_obj.name, status_upload)
                            if gemini_file:
                                # Assign this Gemini file object to all relevant skills
                                if st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
                                    for sk in SKILLS_TO_EVALUATE_EN: gemini_file_objects[sk] = gemini_file
                                else: # MODE_SINGLE_VIDEO_ONE_SKILL
                                    gemini_file_objects[skill_key_to_process] = gemini_file
                            else:
                                upload_successful = False # Flag error if upload fails

                    # Handle multi-video upload
                    elif st.session_state.analysis_mode == MODE_MULTI_VIDEO_AR:
                        for skill_key, uploaded_file_obj in st.session_state.uploaded_files_state.items():
                            if uploaded_file_obj:
                                status_upload.write(f"رفع ومعالجة فيديو: {SKILLS_LABELS_AR[skill_key]}...")
                                # Save to temp local file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1]) as tmp_file:
                                    tmp_file.write(uploaded_file_obj.getvalue())
                                    video_path = tmp_file.name
                                    local_temp_files[skill_key] = video_path
                                # Upload to Gemini and wait
                                gemini_file = upload_and_wait_gemini(video_path, uploaded_file_obj.name, status_upload)
                                if gemini_file:
                                    gemini_file_objects[skill_key] = gemini_file
                                else:
                                    upload_successful = False
                                    break # Stop further uploads if one fails
                            else:
                                st.error(f"ملف مفقود للمهارة: {SKILLS_LABELS_AR[skill_key]}!")
                                upload_successful = False
                                break

                    if not upload_successful:
                        status_upload.update(label="❌ فشل رفع أو معالجة بعض الفيديوهات.", state="error", expanded=True)
                    else:
                        status_upload.update(label="✅ اكتمل رفع ومعالجة جميع الفيديوهات المطلوبة.", state="complete", expanded=False)

                except Exception as e_upload:
                    st.error(f"❌ حدث خطأ فادح أثناء رفع الفيديوهات: {e_upload}")
                    logging.error(f"Fatal error during video upload phase: {e_upload}", exc_info=True)
                    upload_successful = False
                    status_upload.update(label="❌ خطأ فادح في الرفع", state="error")

            # --- Step 2: Analyze Videos (only if upload was successful) ---
            if upload_successful and gemini_file_objects:
                with st.status("🧠 تحليل المهارات بواسطة Gemini...", expanded=True) as status_analysis:
                    try:
                        skills_to_process_keys = list(gemini_file_objects.keys())
                        for skill_key in skills_to_process_keys:
                            gemini_file = gemini_file_objects[skill_key]
                            if gemini_file:
                                status_skill_analysis = st.empty() # Placeholder for this skill's analysis status
                                score = analyze_video_with_prompt(gemini_file, skill_key, status_skill_analysis)
                                results_dict[skill_key] = score
                                if score == 0: # Log warning for zero score
                                    logging.warning(f"Score for {skill_key} is 0. Video: {gemini_file.name}")
                            else:
                                st.warning(f"فشل الحصول على ملف Gemini جاهز للمهارة: {SKILLS_LABELS_AR[skill_key]}")
                                results_dict[skill_key] = 0 # Assign 0 if analysis couldn't run

                        status_analysis.update(label="✅ اكتمل تحليل Gemini لجميع المهارات المطلوبة!", state="complete", expanded=False)
                        # Calculate final grade only if all skills were analyzed (modes 1 & 2)
                        if st.session_state.analysis_mode != MODE_SINGLE_VIDEO_ONE_SKILL_AR:
                            if len(results_dict) == len(SKILLS_TO_EVALUATE_EN):
                                st.session_state.evaluation_results = evaluate_final_grade_from_individual_scores(results_dict)
                                st.success("🎉 تم حساب التقييم النهائي بنجاح!")
                                st.balloons()
                            else:
                                st.warning("لم يتم تحليل جميع المهارات، التقييم النهائي قد يكون غير مكتمل.")
                                st.session_state.evaluation_results = {"scores": results_dict, "grade": "غير مكتمل", "total_score": sum(results_dict.values()), "max_score": "N/A"}
                        else: # Single skill mode
                            st.session_state.evaluation_results = {"scores": results_dict, "grade": "N/A", "total_score": sum(results_dict.values()), "max_score": MAX_SCORE_PER_SKILL}
                            st.success(f"🎉 اكتمل تحليل مهارة '{SKILLS_LABELS_AR.get(list(results_dict.keys())[0], '')}'!")


                    except Exception as e_analyze:
                        st.error(f"❌ حدث خطأ فادح أثناء مرحلة التحليل: {e_analyze}")
                        logging.error(f"Fatal error during analysis phase: {e_analyze}", exc_info=True)
                        error_occurred = True
                        status_analysis.update(label="❌ فشل التحليل", state="error")
                        st.session_state.evaluation_results = None # Clear results on fatal error

            # --- Step 3: Cleanup ---
        finally:
            status_global.info("🧹 جاري تنظيف الملفات المؤقتة...")
            logging.info("Starting cleanup phase...")
            # Delete local temp files
            for skill, path in local_temp_files.items():
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logging.info(f"Deleted local temp file: {path}")
                    except Exception as e_del_local:
                        logging.warning(f"Could not delete local temp file {path}: {e_del_local}")

            # Delete Gemini cloud files (handle potential duplicates in single video mode)
            deleted_cloud_files = set()
            for skill_key, gemini_file in gemini_file_objects.items():
                 if gemini_file and gemini_file.name not in deleted_cloud_files:
                     delete_gemini_file(gemini_file, status_global)
                     deleted_cloud_files.add(gemini_file.name)
            status_global.empty() # Clear cleanup message
            logging.info("Cleanup phase finished.")


    # --- Display Stored Results ---
    if st.session_state.evaluation_results:
        results = st.session_state.evaluation_results
        st.markdown("---")
        st.markdown("### 🏆 النتائج النهائية (بناءً على تحليل Gemini) 🏆")

        # Display differently based on mode
        if st.session_state.analysis_mode != MODE_SINGLE_VIDEO_ONE_SKILL_AR:
            # Full evaluation modes
            res_col1, res_col2 = st.columns(2)
            with res_col1: st.metric("🎯 التقدير العام", results['grade'])
            with res_col2: st.metric("📊 مجموع النقاط", f"{results['total_score']} / {results['max_score']}")
            try:
                st.write("📈 رسم بياني للدرجات:")
                plot_fig = plot_results(results)
                st.pyplot(plot_fig)
                plt.clf(); plt.close(plot_fig) # Clear figure
            except Exception as plot_err:
                 st.error(f"حدث خطأ أثناء إنشاء الرسم البياني: {plot_err}")
                 logging.error(f"Plotting failed: {plot_err}", exc_info=True)
                 st.write(results['scores']) # Fallback
        else:
            # Single skill mode
            if results['scores']:
                skill_key_analyzed = list(results['scores'].keys())[0]
                skill_label_analyzed = SKILLS_LABELS_AR.get(skill_key_analyzed, skill_key_analyzed)
                score_analyzed = results['scores'][skill_key_analyzed]
                st.metric(f"🏅 نتيجة مهارة '{skill_label_analyzed}'", f"{score_analyzed} / {MAX_SCORE_PER_SKILL}")
            else:
                 st.warning("لم يتم العثور على نتائج لعرضها.")


# ==================================
# ==    Other Pages (Placeholders - Arabic) ==
# ==================================
elif st.session_state.page == 'نجم_لا_يغيب':
    st.markdown("---")
    st.markdown("## ⭐ نجم لا يغيب ⭐")
    st.info("سيتم استخدام Gemini API لتحليل جوانب أخرى من أداء اللاعب في هذه الميزة.")
    # TODO: Add UI and logic for this page using Gemini

elif st.session_state.page == 'الشخص_المناسب':
    st.markdown("---")
    st.markdown("## ✔️ الشخص المناسب في المكان المناسب ✔️")
    st.info("سيتم استخدام Gemini API لتحليل مجموعة بيانات (سيتم تحديدها لاحقاً) في هذه الميزة.")
    # TODO: Add UI and logic for this page using Gemini

# --- Footer ---
st.markdown("---")
st.caption("AI League - Scout Eye v1.1 (Gemini Powered - عربي) | بدعم من Google Gemini API")

def test_gemini_connection():
    """Test basic Gemini API connectivity with a simple text prompt."""
    try:
        test_prompt = "Please respond with the number 5 to test API connectivity."
        test_response = model.generate_content(test_prompt)
        st.success(f"✅ Gemini API test successful. Response: {test_response.text}")
        logging.info(f"API test successful. Raw response: {test_response}")
        return True
    except Exception as e:
        st.error(f"❌ Gemini API test failed: {e}")
        logging.error(f"API test failed: {e}", exc_info=True)
        return False

# Call this early in your app
if st.button("Test API Connection"):
    test_result = test_gemini_connection()