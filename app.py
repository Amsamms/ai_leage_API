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

# Age Groups
AGE_GROUP_5_8 = "5 إلى 8 سنوات"
AGE_GROUP_8_PLUS = "8 سنوات وأكثر"

# --- Skills for Age Group: 5 to 8 Years ---
SKILLS_AGE_5_8_EN = [
    "Running_Basic", "Ball_Feeling", "Focus_On_Task", "First_Touch_Simple"
]
SKILLS_LABELS_AGE_5_8_AR = {
    "Running_Basic": "الجري",
    "Ball_Feeling": "الإحساس بالكرة",
    "Focus_On_Task": "التركيز وتنفيذ المطلوب", # Rephrased for observability
    "First_Touch_Simple": "اللمسة الأولى (استلام بسيط)" # Clarified
}

# --- Skills for Age Group: 8 Years and Older ---
SKILLS_AGE_8_PLUS_EN = [
    "Jumping", "Running_Control", "Passing", "Receiving", "Zigzag"
]
SKILLS_LABELS_AGE_8_PLUS_AR = {
    "Jumping": "القفز بالكرة (تنطيط الركبة)",
    "Running_Control": "الجري بالكرة (التحكم)",
    "Passing": "التمرير",
    "Receiving": "استقبال الكرة",
    "Zigzag": "المراوغة (زجزاج)"
}

# --- General Constants ---
MAX_SCORE_PER_SKILL = 5
MODEL_NAME = "models/gemini-1.5-pro"

# --- Analysis Modes (Simplified - Arabic) ---
MODE_SINGLE_VIDEO_ALL_SKILLS_AR = "تقييم جميع مهارات الفئة العمرية (فيديو واحد)"
MODE_SINGLE_VIDEO_ONE_SKILL_AR = "تقييم مهارة محددة (فيديو واحد)"

# --- Gemini API Configuration ---
try:
    # Load API key from Streamlit secrets
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
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
    """Loads the Gemini model with specific configurations."""
    try:
        generation_config = {
             "temperature": 0.1, # Low temp for consistent scoring
             "top_p": 1,
             "top_k": 1,
             "max_output_tokens": 50, # Enough for just the score
        }
        # Using minimal safety settings - USE WITH CAUTION
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logging.info(f"Gemini Model '{MODEL_NAME}' loaded with MINIMUM safety settings (BLOCK_NONE).")
        return model
    except Exception as e:
        st.error(f"❗️ فشل تحميل نموذج Gemini '{MODEL_NAME}': {e}")
        logging.error(f"Gemini model loading failed: {e}")
        return None

model = load_gemini_model()
if not model:
    st.stop()

# --- CSS Styling (Remains the same as previous version) ---
st.markdown("""
<style>
    /* ... (Existing CSS styles - no changes needed here) ... */
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

def create_prompt_for_skill(skill_key_en, age_group):
    """Generates a specific prompt WITH a detailed rubric based on age group."""

    specific_rubric = "لا توجد معايير محددة لهذه المهارة في هذه الفئة العمرية." # Default
    skill_name_ar = skill_key_en # Default

    # --- Rubrics for Age Group: 5 to 8 Years ---
    if age_group == AGE_GROUP_5_8:
        skill_name_ar = SKILLS_LABELS_AGE_5_8_AR.get(skill_key_en, skill_key_en)
        rubrics_5_8 = {
            "Running_Basic": """
            **معايير تقييم الجري (5-8 سنوات):**
            - 0: لا يستطيع الجري أو يمشي فقط.
            - 1: يجري بشكل غير متزن أو بطيء جدًا.
            - 2: يجري بوتيرة مقبولة ولكن ببعض التعثر أو التردد.
            - 3: يجري بثقة وتوازن جيدين لمعظم المسافة.
            - 4: يجري بسرعة جيدة وتوازن ممتاز.
            - 5: يجري بسرعة عالية وتناسق حركي ممتاز وواضح.
            """,
            "Ball_Feeling": """
            **معايير تقييم الإحساس بالكرة (5-8 سنوات):**
            - 0: يتجنب لمس الكرة أو يفقدها فورًا عند اللمس.
            - 1: يلمس الكرة بقدم واحدة فقط بشكل متردد، الكرة تبتعد كثيرًا.
            - 2: يحاول لمس الكرة بكلتا القدمين، لكن التحكم ضعيف.
            - 3: يظهر بعض التحكم الأساسي، يبقي الكرة قريبة أحيانًا.
            - 4: يظهر تحكمًا جيدًا، يلمس الكرة بباطن وظاهر القدم، يحافظ عليها قريبة نسبيًا.
            - 5: يظهر تحكمًا ممتازًا ولمسات واثقة ومتنوعة، يبقي الكرة قريبة جدًا أثناء الحركة البسيطة.
            """,
            "Focus_On_Task": """
            **معايير تقييم التركيز وتنفيذ المطلوب (5-8 سنوات):** (يُقيّم بناءً على السلوك المُلاحظ في الفيديو المتعلق بالمهمة الكروية الظاهرة)
            - 0: لا يُظهر أي اهتمام بالمهمة الكروية، يتشتت تمامًا.
            - 1: يبدأ المهمة لكن يتشتت بسرعة وبشكل متكرر.
            - 2: يحاول إكمال المهمة لكن يفتقر للتركيز المستمر، يتوقف أو ينظر حوله كثيرًا.
            - 3: يركز بشكل مقبول على المهمة، يكمل أجزاء منها بانتباه.
            - 4: يظهر تركيزًا جيدًا ومستمرًا على المهمة الكروية المعروضة في الفيديو.
            - 5: يظهر تركيزًا عاليًا وانغماسًا واضحًا في المهمة الكروية، يحاول بجدية وإصرار.
            """,
            "First_Touch_Simple": """
            **معايير تقييم اللمسة الأولى (استلام بسيط) (5-8 سنوات):**
            - 0: الكرة ترتد بعيدًا جدًا عن السيطرة عند أول لمسة.
            - 1: يوقف الكرة بصعوبة، تتطلب لمسات متعددة للسيطرة.
            - 2: يستلم الكرة بشكل مقبول لكنها تبتعد قليلاً، يتطلب خطوة إضافية للتحكم.
            - 3: استلام جيد، اللمسة الأولى تبقي الكرة ضمن نطاق قريب.
            - 4: استلام جيد جدًا، لمسة أولى نظيفة تهيئ الكرة أمامه مباشرة.
            - 5: استلام ممتاز، لمسة أولى ناعمة وواثقة، سيطرة فورية.
            """
        }
        specific_rubric = rubrics_5_8.get(skill_key_en, specific_rubric)

    # --- Rubrics for Age Group: 8 Years and Older ---
    elif age_group == AGE_GROUP_8_PLUS:
        skill_name_ar = SKILLS_LABELS_AGE_8_PLUS_AR.get(skill_key_en, skill_key_en)
        rubrics_8_plus = {
             "Jumping": """
             **معايير تقييم القفز بالكرة (تنطيط الركبة) (8+ سنوات):**
             - 0: لا توجد محاولات أو لمسات ناجحة بالركبة أثناء الطيران.
             - 1: لمسة واحدة ناجحة بالركبة أثناء الطيران، مع تحكم ضعيف.
             - 2: لمستان ناجحتان بالركبة أثناء الطيران، تحكم مقبول.
             - 3: ثلاث لمسات ناجحة بالركبة، تحكم جيد وثبات.
             - 4: أربع لمسات ناجحة، تحكم ممتاز وثبات هوائي جيد.
             - 5: خمس لمسات أو أكثر، تحكم استثنائي، إيقاع وثبات ممتازين.
             """,
             "Running_Control": """
             **معايير تقييم الجري بالكرة (التحكم) (8+ سنوات):**
             - 0: تحكم ضعيف جدًا، الكرة تبتعد كثيرًا عن القدم.
             - 1: تحكم ضعيف، الكرة تبتعد بشكل ملحوظ أحيانًا.
             - 2: تحكم مقبول، الكرة تبقى ضمن نطاق واسع حول اللاعب.
             - 3: تحكم جيد، الكرة تبقى قريبة بشكل عام أثناء الجري بسرعات مختلفة.
             - 4: تحكم جيد جدًا، الكرة قريبة باستمرار حتى مع تغيير السرعة والاتجاه البسيط.
             - 5: تحكم ممتاز، الكرة تبدو ملتصقة بالقدم، سيطرة كاملة حتى مع المناورات.
             """,
             "Passing": """
             **معايير تقييم التمرير (8+ سنوات):**
             - 0: تمريرة خاطئة تمامًا أو ضعيفة جدًا أو بدون دقة.
             - 1: تمريرة بدقة ضعيفة أو قوة غير مناسبة بشكل كبير.
             - 2: تمريرة مقبولة تصل للهدف ولكن بقوة أو دقة متوسطة.
             - 3: تمريرة جيدة ودقيقة بقوة مناسبة للمسافة والهدف.
             - 4: تمريرة دقيقة جدًا ومتقنة بقوة مثالية، تضع المستلم في وضع جيد.
             - 5: تمريرة استثنائية، دقة وقوة وتوقيت مثالي، تكسر الخطوط أو تضع المستلم في موقف ممتاز.
             """,
             "Receiving": """
             **معايير تقييم استقبال الكرة (8+ سنوات):**
             - 0: فشل في السيطرة على الكرة تمامًا عند الاستقبال.
             - 1: لمسة أولى سيئة، الكرة تبتعد كثيرًا أو تتطلب جهدًا للسيطرة عليها.
             - 2: استقبال مقبول، الكرة تحت السيطرة بعد لمستين أو بحركة إضافية.
             - 3: استقبال جيد، لمسة أولى نظيفة تبقي الكرة قريبة ومتاحة للعب.
             - 4: استقبال جيد جدًا، لمسة أولى ممتازة تهيئ الكرة للخطوة التالية بسهولة (تمرير، تسديد، مراوغة).
             - 5: استقبال استثنائي، لمسة أولى مثالية تحت الضغط، تحكم فوري وسلس، يسمح باللعب السريع.
             """,
             "Zigzag": """
             **معايير تقييم المراوغة (زجزاج) (8+ سنوات):**
             - 0: فقدان السيطرة على الكرة عند محاولة تغيير الاتجاه بين الأقماع.
             - 1: تغيير اتجاه بطيء مع ابتعاد الكرة عن القدم بشكل واضح.
             - 2: تغيير اتجاه مقبول مع الحفاظ على الكرة ضمن نطاق تحكم واسع، يلمس الأقماع أحيانًا.
             - 3: تغيير اتجاه جيد مع إبقاء الكرة قريبة نسبيًا، يتجنب الأقماع.
             - 4: تغيير اتجاه سريع وسلس مع إبقاء الكرة قريبة جدًا من القدم.
             - 5: تغيير اتجاه خاطف وسلس مع سيطرة تامة على الكرة (تبدو ملتصقة بالقدم)، وخفة حركة واضحة.
             """
        }
        specific_rubric = rubrics_8_plus.get(skill_key_en, specific_rubric)

    # --- Construct the Final Prompt ---
    prompt = f"""
    مهمتك هي تقييم مهارة كرة القدم '{skill_name_ar}' المعروضة في الفيديو للاعب ضمن الفئة العمرية '{age_group}'.
    استخدم المعايير التالية **حصراً** لتقييم الأداء وتحديد درجة رقمية من 0 إلى {MAX_SCORE_PER_SKILL}:

    {specific_rubric}

    شاهد الفيديو بعناية. بناءً على المعايير المذكورة أعلاه فقط، ما هي الدرجة التي تصف أداء اللاعب بشكل أفضل؟

    هام جدًا: قم بالرد بالدرجة الرقمية الصحيحة فقط (مثال: "3" أو "5"). لا تقم بتضمين أي شروحات أو أوصاف أو أي نص آخر أو رموز إضافية. فقط الرقم.
    """
    return prompt

def upload_and_wait_gemini(video_path, display_name="video_upload", status_placeholder=st.empty()):
    """Uploads video, waits for ACTIVE state, returns file object or None."""
    uploaded_file = None
    status_placeholder.info(f"⏳ جاري رفع الفيديو '{os.path.basename(display_name)}'...") # Use display name
    logging.info(f"Starting upload for {display_name}")
    try:
        # Use a unique name for the upload based on time to avoid potential conflicts
        safe_display_name = f"upload_{int(time.time())}_{os.path.basename(display_name)}"
        uploaded_file = genai.upload_file(path=video_path, display_name=safe_display_name)
        status_placeholder.info(f"📤 اكتمل الرفع لـ '{display_name}'. برجاء الانتظار للمعالجة بواسطة Google...")
        logging.info(f"Upload API call successful for {display_name}, file name: {uploaded_file.name}. Waiting for ACTIVE state.")

        # Increased timeout for potentially longer videos/processing
        timeout = 300
        start_time = time.time()
        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                logging.error(f"Timeout waiting for file processing for {uploaded_file.name} ({display_name})")
                raise TimeoutError(f"انتهت مهلة معالجة الفيديو '{display_name}'. حاول مرة أخرى أو استخدم فيديو أقصر.")
            time.sleep(15) # Check less frequently
            uploaded_file = genai.get_file(uploaded_file.name)
            logging.debug(f"File {uploaded_file.name} ({display_name}) state: {uploaded_file.state.name}")

        if uploaded_file.state.name == "FAILED":
            logging.error(f"File processing failed for {uploaded_file.name} ({display_name})")
            raise ValueError(f"فشلت معالجة الفيديو '{display_name}' من جانب Google.")
        elif uploaded_file.state.name != "ACTIVE":
             logging.error(f"Unexpected file state {uploaded_file.state.name} for {uploaded_file.name} ({display_name})")
             raise ValueError(f"حالة ملف فيديو غير متوقعة: {uploaded_file.state.name} لـ '{display_name}'.")

        status_placeholder.success(f"✅ الفيديو '{display_name}' جاهز للتحليل.")
        logging.info(f"File {uploaded_file.name} ({display_name}) is ACTIVE.")
        return uploaded_file

    except Exception as e:
        status_placeholder.error(f"❌ خطأ أثناء رفع/معالجة الفيديو لـ '{display_name}': {e}")
        logging.error(f"Upload/Wait failed for '{display_name}': {e}", exc_info=True)
        # Attempt cleanup if upload started but failed processing
        if uploaded_file and uploaded_file.state.name != "ACTIVE":
            try:
                logging.warning(f"Attempting to delete potentially failed/stuck file: {uploaded_file.name} ({display_name})")
                genai.delete_file(uploaded_file.name)
                logging.info(f"Cleaned up failed/stuck file: {uploaded_file.name}")
            except Exception as del_e:
                 logging.warning(f"Failed to delete file {uploaded_file.name} after upload error: {del_e}")
        return None


def analyze_video_with_prompt(gemini_file_obj, skill_key_en, age_group, status_placeholder=st.empty()):
    """
    Analyzes an ACTIVE video file object with a specific skill prompt for the given age group,
    includes debugging output, and handles potential empty/blocked responses.
    """
    score = 0 # Default score

    # Determine the correct Arabic skill name based on age group
    if age_group == AGE_GROUP_5_8:
        skill_name_ar = SKILLS_LABELS_AGE_5_8_AR.get(skill_key_en, skill_key_en)
    elif age_group == AGE_GROUP_8_PLUS:
        skill_name_ar = SKILLS_LABELS_AGE_8_PLUS_AR.get(skill_key_en, skill_key_en)
    else:
        skill_name_ar = skill_key_en # Fallback

    prompt = create_prompt_for_skill(skill_key_en, age_group)

    status_placeholder.info(f"🧠 Gemini يحلل الآن مهارة '{skill_name_ar}' للفئة العمرية '{age_group}'...")
    logging.info(f"Requesting analysis for skill '{skill_key_en}' (Age: {age_group}) using file {gemini_file_obj.name}")
    logging.debug(f"Prompt for {skill_key_en} (Age: {age_group}):\n{prompt}") # Log the prompt

    try:
        # Increased timeout for analysis
        response = model.generate_content([prompt, gemini_file_obj], request_options={"timeout": 180})

        # --- Optional DEBUG block (keep minimized in production) ---
        # try:
        #     with st.expander(f"🐞 معلومات تصحيح لـ '{skill_name_ar}' (اضغط للتوسيع)", expanded=False):
        #         st.write("**Prompt Feedback:**", response.prompt_feedback)
        #         st.write("**Candidates:**", response.candidates)
        #         logging.info(f"Full Gemini Response Object for {skill_key_en} (Age: {age_group}): {response}")
        # except Exception as debug_e:
        #     logging.warning(f"Error displaying debug info in UI for {skill_key_en}: {debug_e}")
        # --- End Optional DEBUG block ---

        # --- Response Checking & Parsing ---
        prompt_blocked = False
        block_reason_text = "N/A"
        try:
            if response.prompt_feedback.block_reason:
                prompt_blocked = True
                block_reason_text = str(response.prompt_feedback.block_reason)
                st.warning(f"⚠️ استجابة Gemini محظورة لـ '{skill_name_ar}' بسبب: {block_reason_text}. النتيجة=0.")
                logging.warning(f"Response blocked (prompt feedback) for {skill_key_en} (Age: {age_group}): {block_reason_text}. File: {gemini_file_obj.name}")
        except AttributeError:
            logging.debug(f"No block_reason found in prompt_feedback for {skill_key_en} (Age: {age_group}).")
        except Exception as feedback_err:
             st.warning(f"⚠️ خطأ غير متوقع في الوصول إلى prompt_feedback لـ '{skill_name_ar}': {feedback_err}.")
             logging.warning(f"Unexpected error accessing prompt_feedback for {skill_key_en} (Age: {age_group}): {feedback_err}.")

        if prompt_blocked or not response.candidates:
             if not response.candidates and not prompt_blocked:
                 st.warning(f"⚠️ استجابة Gemini فارغة (لا توجد مرشحات) لـ '{skill_name_ar}'. النتيجة=0.")
                 logging.warning(f"Response candidates list is empty for {skill_key_en} (Age: {age_group}). File: {gemini_file_obj.name}")
             score = 0
        else:
            try:
                raw_score_text = response.text.strip()
                logging.info(f"Gemini Raw Response Text for {skill_key_en} (Age: {age_group}): '{raw_score_text}'")

                # Use regex to find the first sequence of digits in the response
                match = re.search(r'\d+', raw_score_text)
                if match:
                    parsed_score = int(match.group(0))
                    # Clamp score to the expected range
                    score = max(0, min(MAX_SCORE_PER_SKILL, parsed_score))
                    status_placeholder.success(f"✅ اكتمل تحليل '{skill_name_ar}'. النتيجة: {score}")
                    logging.info(f"Analysis for {skill_key_en} (Age: {age_group}) successful. Raw: '{raw_score_text}', Parsed Score: {parsed_score}, Final Score: {score}. File: {gemini_file_obj.name}")
                    if parsed_score != score:
                         logging.warning(f"Score for {skill_key_en} (Age: {age_group}) was clamped from {parsed_score} to {score}.")
                else:
                     st.warning(f"⚠️ لم يتم العثور على رقم في استجابة Gemini لـ '{skill_name_ar}' ('{raw_score_text}'). النتيجة=0.")
                     logging.warning(f"Could not parse score (no digits found) for {skill_key_en} (Age: {age_group}) from text: '{raw_score_text}'. File: {gemini_file_obj.name}")
                     score = 0

            except (ValueError, AttributeError) as e_parse:
                 # Try to get fallback text for logging if .text failed or int() failed
                 candidate_text_fallback = "Error retrieving/parsing text"
                 try:
                     if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                         candidate_text_fallback = response.candidates[0].content.parts[0].text
                 except Exception: pass
                 st.warning(f"⚠️ لم نتمكن من تحليل النتيجة من استجابة Gemini لـ '{skill_name_ar}'. الخطأ: {e_parse}. النص التقريبي: '{candidate_text_fallback}'. النتيجة=0.")
                 logging.warning(f"Could not parse score for {skill_key_en} (Age: {age_group}). Access/Parse Error: {e_parse}. Fallback Text: '{candidate_text_fallback}'. File: {gemini_file_obj.name}. Candidates: {response.candidates}")
                 score = 0
            except Exception as e_generic_parse:
                 st.error(f"⚠️ خطأ غير متوقع أثناء تحليل استجابة Gemini لـ '{skill_name_ar}': {e_generic_parse}. النتيجة=0.")
                 logging.error(f"Unexpected parsing error for {skill_key_en} (Age: {age_group}): {e_generic_parse}. File: {gemini_file_obj.name}", exc_info=True)
                 score = 0
        # --- END RESPONSE CHECKING & PARSING ---

    # --- Outer Exception Handling ---
    except genai.types.generation_types.BlockedPromptException as bpe:
         st.error(f"❌ تم حظر طلب Gemini API نفسه لـ '{skill_name_ar}': {bpe}")
         logging.error(f"Prompt blocked during API call for {skill_key_en} (Age: {age_group}): {bpe}. File: {gemini_file_obj.name}")
         score = 0
    except genai.types.generation_types.StopCandidateException as sce:
         st.error(f"❌ توقف إنشاء الاستجابة بشكل غير متوقع لـ '{skill_name_ar}' (ربما لأسباب تتعلق بالأمان/السياسة): {sce}")
         logging.error(f"Analysis stopped (safety/policy) for {skill_key_en} (Age: {age_group}): {sce}. File: {gemini_file_obj.name}")
         score = 0
    except TimeoutError as te:
         st.error(f"❌ انتهت مهلة استدعاء Gemini API لـ '{skill_name_ar}': {te}")
         logging.error(f"Timeout during API call/generation for {skill_key_en} (Age: {age_group}): {te}. File: {gemini_file_obj.name}")
         score = 0
    except Exception as e:
        st.error(f"❌ حدث خطأ عام أثناء استدعاء Gemini API لـ '{skill_name_ar}': {e}")
        logging.error(f"General Gemini analysis failed for {skill_key_en} (Age: {age_group}): {e}. File: {gemini_file_obj.name}", exc_info=True)
        score = 0

    return score


def delete_gemini_file(gemini_file_obj, status_placeholder=st.empty()):
    """Deletes the uploaded file from Gemini Cloud Storage."""
    if not gemini_file_obj: return
    try:
        # Use the unique name for logging/status, but the actual name for deletion
        display_name = gemini_file_obj.display_name # Should contain the unique upload name
        status_placeholder.info(f"🗑️ جاري حذف الملف المرفوع '{display_name}' من التخزين السحابي...")
        logging.info(f"Attempting to delete cloud file: {gemini_file_obj.name} (Display: {display_name})")
        genai.delete_file(gemini_file_obj.name)
        # Do not clear the placeholder immediately, let the calling function manage it
        # status_placeholder.empty()
        logging.info(f"Cloud file deleted successfully: {gemini_file_obj.name} (Display: {display_name})")
    except Exception as e:
        # Display warning but allow process to continue
        st.warning(f"⚠️ لم نتمكن من حذف الملف السحابي {gemini_file_obj.name} (Display: {display_name}): {e}")
        logging.warning(f"Could not delete cloud file {gemini_file_obj.name} (Display: {display_name}): {e}")


# =========== Grading and Plotting Functions =================

def evaluate_final_grade_from_individual_scores(scores_dict):
    """Calculates total score, max score, percentage, and grade."""
    if not scores_dict:
        return {"scores": {}, "total_score": 0, "grade": "N/A", "max_score": 0}

    total = sum(scores_dict.values())
    max_possible = len(scores_dict) * MAX_SCORE_PER_SKILL
    percentage = (total / max_possible) * 100 if max_possible > 0 else 0

    # Define grades (adjust thresholds if needed)
    if percentage >= 90: grade = 'ممتاز (A)'
    elif percentage >= 75: grade = 'جيد جداً (B)'
    elif percentage >= 55: grade = 'جيد (C)'
    elif percentage >= 40: grade = 'مقبول (D)'
    else: grade = 'ضعيف (F)'

    return {"scores": scores_dict, "total_score": total, "grade": grade, "max_score": max_possible}

def plot_results(results, skills_labels_ar):
    """Generates and returns a matplotlib figure of the scores with correct Arabic labels."""
    if not results or 'scores' not in results or not results['scores']:
        logging.warning("Plotting attempted with invalid or empty results.")
        # Return an empty figure or some indicator
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, get_display(arabic_reshaper.reshape("لا توجد بيانات لعرضها")),
                ha='center', va='center', color='white')
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))
        ax.axis('off')
        return fig

    scores_dict = results['scores']
    # Ensure we only plot skills that are in the provided labels dictionary
    valid_keys_en = [key for key in scores_dict.keys() if key in skills_labels_ar]
    if not valid_keys_en:
         logging.warning("No matching keys found between results and skills_labels_ar for plotting.")
         fig, ax = plt.subplots()
         ax.text(0.5, 0.5, get_display(arabic_reshaper.reshape("خطأ: عدم تطابق بيانات الرسم")),
                 ha='center', va='center', color='white')
         fig.patch.set_alpha(0)
         ax.set_facecolor((0, 0, 0, 0))
         ax.axis('off')
         return fig

    # Prepare labels and scores using only valid keys
    try:
        reshaped_labels = [get_display(arabic_reshaper.reshape(skills_labels_ar[key_en])) for key_en in valid_keys_en]
        scores = [scores_dict[key_en] for key_en in valid_keys_en]
        # Handle grade display, might be 'N/A' for single skill
        grade_display = results.get('grade', 'N/A')
        if grade_display != 'N/A':
            plot_title_text = f"التقييم النهائي - التقدير: {grade_display} ({results.get('total_score', 0)}/{results.get('max_score', 0)})"
        else:
            # Single skill mode title or incomplete analysis
            plot_title_text = "نتيجة المهارة"
            if len(valid_keys_en) == 1:
                 plot_title_text = f"نتيجة مهارة: {reshaped_labels[0]}"


        plot_title = get_display(arabic_reshaper.reshape(plot_title_text))
        y_axis_label = get_display(arabic_reshaper.reshape(f"الدرجة (من {MAX_SCORE_PER_SKILL})"))
    except Exception as e:
        st.warning(f"حدث خطأ أثناء تهيئة نص الرسم البياني العربي: {e}")
        logging.warning(f"Arabic reshaping/label preparation failed for plot: {e}")
        # Fallback to English keys if reshaping fails
        reshaped_labels = valid_keys_en
        scores = [scores_dict[key_en] for key_en in valid_keys_en]
        plot_title = f"Evaluation - Grade: {results.get('grade','N/A')} ({results.get('total_score',0)}/{results.get('max_score',0)})"
        y_axis_label = f"Score (out of {MAX_SCORE_PER_SKILL})"


    fig, ax = plt.subplots(figsize=(max(6, len(scores)*1.5), 6)) # Dynamic width
    bars = ax.bar(reshaped_labels, scores)
    ax.set_ylim(0, MAX_SCORE_PER_SKILL + 0.5)
    ax.set_ylabel(y_axis_label, fontsize=12, fontweight='bold', color='white')
    ax.set_title(plot_title, fontsize=14, fontweight='bold', color='white')

    # Color bars based on score
    colors = ['#2ca02c' if s >= 4 else '#ff7f0e' if s >= 2.5 else '#d62728' for s in scores] # Green, Orange, Red
    for bar, color in zip(bars, colors): bar.set_color(color)

    # Add score labels on top of bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval}', ha='center', va='bottom', fontsize=11, color='white', fontweight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
    ax.tick_params(axis='x', labelsize=11, rotation=15, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # Transparent background
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0)) # Fully transparent axes background

    plt.tight_layout()
    return fig


# =========== Streamlit App Layout (Arabic) ====================================

# Initialize session state variables
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = None
# Remove multi-video mode, set a default valid mode
if 'analysis_mode' not in st.session_state: st.session_state.analysis_mode = MODE_SINGLE_VIDEO_ALL_SKILLS_AR
if 'selected_skill_key' not in st.session_state: st.session_state.selected_skill_key = None # Store English key
if 'selected_age_group' not in st.session_state: st.session_state.selected_age_group = AGE_GROUP_8_PLUS # Default age
if 'uploaded_file_state' not in st.session_state: st.session_state.uploaded_file_state = None # Store single uploaded file object
if 'gemini_file_object' not in st.session_state: st.session_state.gemini_file_object = None # Store the Gemini file object across runs if needed

# --- Top Row: Logo ---
col1, col_mid, col2 = st.columns([1, 3, 1])
with col1:
    st.markdown("<p style='font-size: 1.2em; font-weight: bold;'>AI LEAGUE</p>", unsafe_allow_html=True)
# Optional: Add logo image if you have one
# with col1:
#     st.image("path/to/your/logo.png", width=100)

# --- Center Area: Main Logo, Title, Slogan ---
st.container() # Helps center content
st.markdown("<h1 style='text-align: center; color: white; margin-top: 20px;'>Scout Eye</h1>", unsafe_allow_html=True)
st.markdown("<div class='title-text'>عين الكشاف</div>", unsafe_allow_html=True)
st.markdown("<div class='slogan-text'>نكتشف ، نحمي ، ندعم</div>", unsafe_allow_html=True)

# --- Bottom Area: Clickable Options (Arabic) ---
st.container()
col_b1, col_b2, col_b3 = st.columns(3)
button_keys = ["btn_person", "btn_star", "btn_legend"]

# Reset results when switching main pages
if col_b1.button("✔️ الشخص المناسب", key=button_keys[0]):
    st.session_state.page = 'الشخص_المناسب'
    st.session_state.evaluation_results = None
    st.session_state.uploaded_file_state = None
    st.session_state.gemini_file_object = None # Clear Gemini file object too
if col_b2.button("⭐ نجم لا يغيب", key=button_keys[1]):
    st.session_state.page = 'نجم_لا_يغيب'
    st.session_state.evaluation_results = None
    st.session_state.uploaded_file_state = None
    st.session_state.gemini_file_object = None
if col_b3.button("⚽ إسطورة الغد", key=button_keys[2]):
    st.session_state.page = 'اسطورة_الغد'
    st.session_state.evaluation_results = None
    st.session_state.uploaded_file_state = None
    st.session_state.gemini_file_object = None

# --- Conditional Page Content ---

# ==================================
# ==      إسطورة الغد Page       ==
# ==================================
if st.session_state.page == 'اسطورة_الغد':
    st.markdown("---")
    st.markdown("## ⚽ إسطورة الغد - تحليل المهارات بواسطة Gemini ⚽")

    # --- 1. Age Group Selection ---
    st.markdown("<h3 style='text-align: center;'>1. اختر الفئة العمرية للموهبة</h3>", unsafe_allow_html=True)
    age_options = [AGE_GROUP_5_8, AGE_GROUP_8_PLUS]
    st.session_state.selected_age_group = st.radio(
        "الفئة العمرية:",
        options=age_options,
        index=age_options.index(st.session_state.selected_age_group), # Persist selection
        key="age_group_radio",
        horizontal=True # Display horizontally
    )

    # Determine current skill set based on selected age
    if st.session_state.selected_age_group == AGE_GROUP_5_8:
        current_skills_en = SKILLS_AGE_5_8_EN
        current_skills_labels_ar = SKILLS_LABELS_AGE_5_8_AR
    else: # AGE_GROUP_8_PLUS
        current_skills_en = SKILLS_AGE_8_PLUS_EN
        current_skills_labels_ar = SKILLS_LABELS_AGE_8_PLUS_AR

    # Reset selected skill if the age group changes and the old skill isn't valid
    if 'selected_skill_key' in st.session_state and st.session_state.selected_skill_key not in current_skills_en:
        st.session_state.selected_skill_key = current_skills_en[0] if current_skills_en else None

    st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.3); margin-top: 0.5em; margin-bottom: 1.5em;'>", unsafe_allow_html=True)

    # --- 2. Analysis Mode Selection ---
    st.markdown("<h3 style='text-align: center;'>2. اختر طريقة التحليل</h3>", unsafe_allow_html=True)
    analysis_options = [MODE_SINGLE_VIDEO_ALL_SKILLS_AR, MODE_SINGLE_VIDEO_ONE_SKILL_AR]
    st.session_state.analysis_mode = st.radio(
        "طريقة التحليل:",
        options=analysis_options,
        index=analysis_options.index(st.session_state.analysis_mode), # Persist selection
        key="analysis_mode_radio",
        horizontal=True # Display horizontally
    )

    st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.3); margin-top: 0.5em; margin-bottom: 1.5em;'>", unsafe_allow_html=True)

    # --- 3. File Upload UI ---
    st.markdown("<h3 style='text-align: center;'>3. ارفع ملف الفيديو</h3>", unsafe_allow_html=True)
    uploaded_file = None # Variable to hold the st.file_uploader object
    skill_to_analyze_key_en = None # English key for the selected skill in single-skill mode

    if st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
        st.markdown(f"<p style='text-align: center; font-size: 1.1em;'>لتقييم جميع مهارات فئة '{st.session_state.selected_age_group}' ({len(current_skills_en)} مهارات)</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📂 ارفع فيديو شامل واحد:",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="upload_single_all_unified" # Use a consistent key
            )

    elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ONE_SKILL_AR:
        st.markdown("<p style='text-align: center; font-size: 1.1em;'>لتقييم مهارة واحدة محددة من فيديو</p>", unsafe_allow_html=True)
        col_select, col_upload = st.columns([1, 2])
        with col_select:
             st.session_state.selected_skill_key = st.selectbox(
                 "اختر المهارة:",
                 options=current_skills_en,
                 format_func=lambda key: current_skills_labels_ar.get(key, key), # Show Arabic label
                 index=current_skills_en.index(st.session_state.selected_skill_key) if st.session_state.selected_skill_key in current_skills_en else 0,
                 key="select_single_skill_unified" # Use a consistent key
             )
             skill_to_analyze_key_en = st.session_state.selected_skill_key
             skill_label_for_upload = current_skills_labels_ar.get(skill_to_analyze_key_en, "المحددة")

        with col_upload:
            uploaded_file = st.file_uploader(
                f"📂 ارفع فيديو مهارة '{skill_label_for_upload}'",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="upload_single_one_unified" # Use a consistent key
                )

    # Store the Streamlit uploaded file object in session state if it exists
    if uploaded_file:
        st.session_state.uploaded_file_state = uploaded_file
    else:
        # If the uploader is cleared (file removed by user), reset state
        st.session_state.uploaded_file_state = None
        st.session_state.gemini_file_object = None # Also clear the processed Gemini object

    # Determine if ready to analyze based on mode and file upload
    ready_to_analyze = False
    if st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
        ready_to_analyze = st.session_state.uploaded_file_state is not None
    elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ONE_SKILL_AR:
        ready_to_analyze = st.session_state.uploaded_file_state is not None and skill_to_analyze_key_en is not None

    st.markdown("---")

    # --- 4. Analysis Button ---
    st.markdown("<h3 style='text-align: center;'>4. ابدأ التحليل</h3>", unsafe_allow_html=True)
    # Center the button
    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
    with button_col2:
        if st.button("🚀 بدء التحليل بواسطة Gemini", key="start_gemini_eval_unified", disabled=not ready_to_analyze, use_container_width=True):
            st.session_state.evaluation_results = None # Clear previous results
            local_temp_file_path = None
            analysis_error = False
            # Use the existing Gemini file object if available and the uploaded file hasn't changed
            should_upload = True
            if st.session_state.gemini_file_object and st.session_state.uploaded_file_state and st.session_state.gemini_file_object.display_name.endswith(st.session_state.uploaded_file_state.name):
                 # Basic check: if Gemini object exists and its name matches the current upload name
                 try:
                      # Verify the file is still ACTIVE on Google's side
                      st.info("🔄 التحقق من حالة الفيديو المرفوع سابقاً...")
                      check_file = genai.get_file(st.session_state.gemini_file_object.name)
                      if check_file.state.name == "ACTIVE":
                           st.success("✅ الفيديو المرفوع سابقاً لا يزال جاهزاً.")
                           should_upload = False
                           logging.info(f"Reusing existing ACTIVE Gemini file: {st.session_state.gemini_file_object.name}")
                      else:
                           st.warning(f"⚠️ الفيديو المرفوع سابقاً لم يعد صالحاً (الحالة: {check_file.state.name}). سيتم إعادة الرفع.")
                           logging.warning(f"Previous Gemini file {st.session_state.gemini_file_object.name} no longer ACTIVE (State: {check_file.state.name}). Re-uploading.")
                           # Clean up the invalid old file reference
                           st.session_state.gemini_file_object = None
                 except Exception as e_check:
                      st.warning(f"⚠️ فشل التحقق من الفيديو السابق ({e_check}). سيتم إعادة الرفع.")
                      logging.warning(f"Failed to check status of previous Gemini file {st.session_state.gemini_file_object.name}: {e_check}. Re-uploading.")
                      st.session_state.gemini_file_object = None


            # --- Step 1: Upload and Process Video (if needed) ---
            if should_upload:
                st.session_state.gemini_file_object = None # Ensure old object is cleared
                status_placeholder_upload = st.empty()
                try:
                    # Save to temp local file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file_state.name)[1]) as tmp_file:
                        tmp_file.write(st.session_state.uploaded_file_state.getvalue())
                        local_temp_file_path = tmp_file.name
                        logging.info(f"Saved uploaded file to temporary path: {local_temp_file_path}")

                    # Upload to Gemini and wait
                    st.session_state.gemini_file_object = upload_and_wait_gemini(
                        local_temp_file_path,
                        st.session_state.uploaded_file_state.name, # Use original filename for display
                        status_placeholder_upload
                        )
                    if not st.session_state.gemini_file_object:
                        analysis_error = True # Flag error if upload fails

                except Exception as e_upload:
                    status_placeholder_upload.error(f"❌ حدث خطأ فادح أثناء تحضير الفيديو: {e_upload}")
                    logging.error(f"Fatal error during video preparation/upload: {e_upload}", exc_info=True)
                    analysis_error = True
                finally:
                     # Clean up local temp file immediately after upload attempt
                    if local_temp_file_path and os.path.exists(local_temp_file_path):
                        try:
                            os.remove(local_temp_file_path)
                            logging.info(f"Deleted local temp file: {local_temp_file_path}")
                        except Exception as e_del_local:
                            logging.warning(f"Could not delete local temp file {local_temp_file_path}: {e_del_local}")

            # --- Step 2: Analyze Video(s) ---
            if not analysis_error and st.session_state.gemini_file_object:
                results_dict = {}
                with st.spinner("🧠 Gemini يحلل المهارات المطلوبة..."):
                    try:
                        skills_to_process_keys = []
                        if st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
                            skills_to_process_keys = current_skills_en
                        elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ONE_SKILL_AR:
                            if skill_to_analyze_key_en: # Should exist if ready_to_analyze was True
                                skills_to_process_keys = [skill_to_analyze_key_en]

                        if not skills_to_process_keys:
                             st.error("لم يتم تحديد مهارات للتحليل.")
                             analysis_error = True
                        else:
                             st.info(f"سيتم تحليل {len(skills_to_process_keys)} مهارة...")
                             analysis_status_container = st.container() # Area for individual skill status

                             for skill_key in skills_to_process_keys:
                                 status_skill_analysis = analysis_status_container.empty() # Placeholder for this skill's status
                                 score = analyze_video_with_prompt(
                                     st.session_state.gemini_file_object,
                                     skill_key,
                                     st.session_state.selected_age_group,
                                     status_skill_analysis
                                 )
                                 results_dict[skill_key] = score
                                 if score == 0 and not analysis_error: # Log non-blocking warnings for 0 scores
                                     logging.warning(f"Score received as 0 for skill {skill_key} (Age: {st.session_state.selected_age_group}). Check video/prompt if unexpected.")
                                 # No need to sleep unless hitting rate limits

                             # --- Step 3: Calculate Final Grade and Store Results ---
                             if st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
                                 if len(results_dict) == len(current_skills_en):
                                     st.session_state.evaluation_results = evaluate_final_grade_from_individual_scores(results_dict)
                                     st.success("🎉 تم حساب التقييم النهائي بنجاح!")
                                     st.balloons()
                                 else:
                                     st.warning(f"لم يتم تحليل جميع المهارات المتوقعة ({len(current_skills_en)}). النتائج قد تكون غير مكتملة.")
                                     # Store partial results anyway
                                     st.session_state.evaluation_results = {"scores": results_dict, "grade": "غير مكتمل", "total_score": sum(results_dict.values()), "max_score": len(current_skills_en) * MAX_SCORE_PER_SKILL}
                             elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ONE_SKILL_AR:
                                 # Only one skill expected
                                 if results_dict:
                                      st.session_state.evaluation_results = {
                                          "scores": results_dict,
                                          "grade": "N/A", # No overall grade for single skill
                                          "total_score": sum(results_dict.values()),
                                          "max_score": MAX_SCORE_PER_SKILL
                                      }
                                      analyzed_skill_label = current_skills_labels_ar.get(list(results_dict.keys())[0], '')
                                      st.success(f"🎉 اكتمل تحليل مهارة '{analyzed_skill_label}'!")
                                 else:
                                      st.error("فشل تحليل المهارة المحددة.")
                                      analysis_error = True

                    except Exception as e_analyze:
                        st.error(f"❌ حدث خطأ فادح أثناء مرحلة التحليل: {e_analyze}")
                        logging.error(f"Fatal error during analysis phase: {e_analyze}", exc_info=True)
                        analysis_error = True
                        st.session_state.evaluation_results = None # Clear results on fatal error

            # --- Cleanup Note ---
            # Gemini file deletion is now handled more robustly:
            # - If upload fails, it attempts deletion in upload_and_wait_gemini.
            # - If analysis runs, the file object remains in session state (`st.session_state.gemini_file_object`).
            # - It will be checked/reused or deleted/re-uploaded on the *next* analysis run if the user changes the video or if it becomes inactive.
            # - Consider adding an explicit "Clear Video & Results" button if long-term storage becomes an issue.


    # --- Display Stored Results ---
    if st.session_state.evaluation_results:
        results = st.session_state.evaluation_results
        st.markdown("---")
        st.markdown("### 🏆 النتائج النهائية (بناءً على تحليل Gemini) 🏆")

        # Determine the correct labels for the plot based on the age group when results were generated
        # (We assume results always correspond to the *currently* selected age group for simplicity here,
        # but ideally, you'd store the age group *with* the results)
        plot_labels_ar = current_skills_labels_ar # Use labels for the currently selected age group

        # Display differently based on mode WHEN THE RESULTS WERE GENERATED
        if 'grade' in results and results['grade'] != "N/A" and results['grade'] != "غير مكتمل":
            # Assumed Full evaluation mode results
            res_col1, res_col2 = st.columns(2)
            with res_col1: st.metric("🎯 التقدير العام", results['grade'])
            with res_col2: st.metric("📊 مجموع النقاط", f"{results.get('total_score', '0')} / {results.get('max_score', '0')}")

            st.markdown("#### 📈 رسم بياني للدرجات:")
            try:
                plot_fig = plot_results(results, plot_labels_ar)
                st.pyplot(plot_fig)
                # Clear the figure from memory after displaying
                plt.close(plot_fig)
            except Exception as plot_err:
                 st.error(f"حدث خطأ أثناء إنشاء الرسم البياني: {plot_err}")
                 logging.error(f"Plotting failed: {plot_err}", exc_info=True)
                 # Fallback: display raw scores
                 with st.expander("عرض الدرجات الخام"):
                     for key, score in results.get('scores', {}).items():
                         label = plot_labels_ar.get(key, key)
                         st.write(f"- {label}: {score}/{MAX_SCORE_PER_SKILL}")

        elif 'scores' in results and results['scores']:
            # Assumed Single skill mode results or incomplete results
            if len(results['scores']) == 1:
                skill_key_analyzed = list(results['scores'].keys())[0]
                skill_label_analyzed = plot_labels_ar.get(skill_key_analyzed, skill_key_analyzed)
                score_analyzed = results['scores'][skill_key_analyzed]
                st.metric(f"🏅 نتيجة مهارة '{skill_label_analyzed}'", f"{score_analyzed} / {MAX_SCORE_PER_SKILL}")
                # Optionally show a simple bar for the single skill
                st.markdown("#### 📈 رسم بياني للدرجة:")
                try:
                    plot_fig = plot_results(results, plot_labels_ar)
                    st.pyplot(plot_fig)
                    plt.close(plot_fig)
                except Exception as plot_err:
                    st.error(f"حدث خطأ أثناء إنشاء الرسم البياني للمهارة الواحدة: {plot_err}")
                    logging.error(f"Single skill plotting failed: {plot_err}", exc_info=True)

            else: # Incomplete results from "All Skills" mode
                st.warning("النتائج غير مكتملة.")
                st.metric("📊 مجموع النقاط (غير مكتمل)", f"{results.get('total_score', '0')} / {results.get('max_score', '0')}")
                st.markdown("#### 📈 رسم بياني للدرجات المتوفرة:")
                try:
                    plot_fig = plot_results(results, plot_labels_ar)
                    st.pyplot(plot_fig)
                    plt.close(plot_fig)
                except Exception as plot_err:
                    st.error(f"حدث خطأ أثناء إنشاء الرسم البياني للنتائج غير المكتملة: {plot_err}")
                    logging.error(f"Incomplete results plotting failed: {plot_err}", exc_info=True)
                    with st.expander("عرض الدرجات الخام"):
                        for key, score in results.get('scores', {}).items():
                            label = plot_labels_ar.get(key, key)
                            st.write(f"- {label}: {score}/{MAX_SCORE_PER_SKILL}")
        else:
             st.warning("لم يتم العثور على نتائج صالحة لعرضها.")


# ==================================
# ==    Other Pages (Placeholders - Arabic) ==
# ==================================
elif st.session_state.page == 'نجم_لا_يغيب':
    st.markdown("---")
    st.markdown("## ⭐ نجم لا يغيب ⭐")
    st.info("سيتم استخدام Gemini API لتحليل جوانب أخرى من أداء اللاعب في هذه الميزة (قيد التطوير).")
    # TODO: Add UI and logic for this page using Gemini

elif st.session_state.page == 'الشخص_المناسب':
    st.markdown("---")
    st.markdown("## ✔️ الشخص المناسب في المكان المناسب ✔️")
    st.info("سيتم استخدام Gemini API لتحليل مجموعة بيانات (سيتم تحديدها لاحقاً) في هذه الميزة (قيد التطوير).")
    # TODO: Add UI and logic for this page using Gemini

# --- Footer ---
st.markdown("---")
st.caption("AI League - Scout Eye v1.2 (Gemini Powered - عربي) | بدعم من Google Gemini API")

# --- Optional: Add an API Test Button ---
def test_gemini_connection():
    """Test basic Gemini API connectivity with a simple text prompt."""
    try:
        test_model = load_gemini_model() # Ensure model is loaded
        if not test_model:
            st.error("فشل تحميل النموذج للاختبار.")
            return False
        test_prompt = "Please respond with only the number 5."
        test_response = test_model.generate_content(test_prompt)
        if "5" in test_response.text:
            st.success(f"✅ اختبار اتصال Gemini API ناجح.")
            logging.info(f"API test successful. Response: {test_response.text}")
            return True
        else:
            st.warning(f"⚠️ اختبار اتصال Gemini API يعمل، لكن الاستجابة غير متوقعة: {test_response.text}")
            logging.warning(f"API test connection OK, but response unexpected: {test_response}")
            return True # Connection is ok, response format is the issue
    except Exception as e:
        st.error(f"❌ فشل اختبار اتصال Gemini API: {e}")
        logging.error(f"API test failed: {e}", exc_info=True)
        return False

st.sidebar.button("اختبار اتصال API", on_click=test_gemini_connection)