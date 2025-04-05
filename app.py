import streamlit as st
import google.generativeai as genai
import os
import tempfile
import time
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# --- Page Configuration ---
st.set_page_config(
    page_title="AI League Scout Eye (Gemini)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants ---
SKILLS_TO_EVALUATE = {
    "Jumping": "1️⃣ فيديو القفز بالكرة (Knee Juggling)",
    "Running": "2️⃣ فيديو الجري بالكرة (Running Control)",
    "Passing": "3️⃣ فيديو التمرير (Passing)",
    "Receiving": "4️⃣ فيديو استقبال الكرة (Receiving Control)",
    "Zigzag": "5️⃣ فيديو المراوغة (Zigzag Dribbling)"
}
MAX_SCORE_PER_SKILL = 5
MODEL_NAME = "gemini-1.5-flash-latest" # Using Flash for speed/cost efficiency

# --- Gemini API Configuration ---
try:
    # Load API key from Streamlit secrets
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    st.success("🔑 Gemini API Key loaded successfully from Secrets.")
except KeyError:
    st.error("❗️ Gemini API Key not found in Streamlit Secrets. Please add `GEMINI_API_KEY` to your secrets.")
    st.stop()
except Exception as e:
    st.error(f"❗️ Failed to configure Gemini API: {e}")
    st.stop()

# --- Gemini Model Setup ---
@st.cache_resource
def load_gemini_model():
    try:
        generation_config = {
            "temperature": 0.1, # Low temperature for consistent scoring
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 50, # Expecting just a number
            # "response_mime_type": "text/plain", # Ensure text output
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        st.success(f"✅ Gemini Model '{MODEL_NAME}' loaded.")
        return model
    except Exception as e:
        st.error(f"❗️ Failed to load Gemini model: {e}")
        return None

model = load_gemini_model()
if not model:
    st.stop() # Stop if model loading failed

# --- CSS Styling (same as before) ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1a1a4a; color: white;
    }
    /* Style buttons */
    .stButton>button {
        background-color: transparent; color: white; border: 1px solid transparent;
        padding: 15px 25px; text-align: center; text-decoration: none;
        display: inline-block; font-size: 1.2em; margin: 15px 10px;
        cursor: pointer; transition: background-color 0.3s ease, border-color 0.3s ease;
        font-weight: bold; border-radius: 8px; min-width: 200px;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.4);
    }
    .stButton>button:active { background-color: rgba(255, 255, 255, 0.2); }
    /* Center content */
    .main .block-container {
        padding-top: 1rem; padding-bottom: 2rem; display: flex;
        flex-direction: column; align-items: center;
    }
    /* Title and Slogan */
    .title-text { font-size: 3em; font-weight: bold; color: white; text-align: center; margin-bottom: 0.3em; }
    .slogan-text { font-size: 1.8em; font-weight: bold; color: white; text-align: center; margin-bottom: 1.5em; direction: rtl; }
    /* Section Headers */
     h2 { color: #d8b8d8; text-align: center; margin-top: 1.5em; font-size: 2em; font-weight: bold; }
     h3 { color: white; text-align: center; margin-top: 1em; font-size: 1.5em; }
    /* File uploader label */
    .stFileUploader label { color: white !important; font-size: 1.1em !important; font-weight: bold; text-align: right !important; width: 100%; }
    /* Metric style */
    .stMetric { background-color: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; text-align: center; }
     .stMetric label { font-weight: bold; color: #d8b8d8; }
     .stMetric div[data-testid="metric-value"] { font-size: 2em; font-weight: bold; color: white; }
    /* Plot background */
    img[alt="matplotlib chart"] { background-color: transparent !important; }
</style>
""", unsafe_allow_html=True)

# =========== Gemini Analysis Function ============================

def create_prompt_for_skill(skill_name):
    """Generates a specific prompt for Gemini based on the skill."""
    base_instruction = f"""
    Analyze the provided video focusing ONLY on the football skill: {skill_name}.
    Evaluate the player's performance based on typical criteria for this skill (control, execution, effectiveness).
    Assign a numerical score from 0 to {MAX_SCORE_PER_SKILL}, where 0 is very poor/no attempt and {MAX_SCORE_PER_SKILL} is excellent execution.

    IMPORTANT: Respond with ONLY the integer score (e.g., "3" or "5"). Do not include explanations, descriptions, or any other text. Just the number.
    """

    # Add more specific criteria hints if needed, but keep it concise
    if skill_name == "Jumping":
        prompt = base_instruction + "\nCriteria hint: Focus on controlled knee touches while airborne."
    elif skill_name == "Running":
        prompt = base_instruction + "\nCriteria hint: Focus on keeping the ball close while moving at pace."
    elif skill_name == "Passing":
        prompt = base_instruction + "\nCriteria hint: Focus on accuracy, weight, and technique of the pass shown."
    elif skill_name == "Receiving":
        prompt = base_instruction + "\nCriteria hint: Focus on the first touch, control, and preparation for the next action."
    elif skill_name == "Zigzag":
        prompt = base_instruction + "\nCriteria hint: Focus on close control while changing direction rapidly."
    else:
        prompt = base_instruction # Default if skill name unknown

    return prompt

def get_skill_score_from_gemini(video_path, skill_name, status_placeholder):
    """Uploads video, runs Gemini analysis, parses score, cleans up."""
    score = 0 # Default score in case of error
    uploaded_file = None
    prompt = create_prompt_for_skill(skill_name)
    status_placeholder.info(f"⏳ Uploading video for {skill_name}...")

    try:
        # 1. Upload the video file to Gemini File API
        uploaded_file = genai.upload_file(path=video_path, display_name=f"{skill_name}_video")
        status_placeholder.info(f"📤 Upload complete for {skill_name}. Waiting for processing...")

        # 2. Wait for the file to be processed by Google
        timeout = 180 # Wait up to 3 minutes for processing
        start_time = time.time()
        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                raise TimeoutError("Video processing timed out.")
            time.sleep(5) # Wait 5 seconds before checking state again
            uploaded_file = genai.get_file(uploaded_file.name) # Refresh file state

        if uploaded_file.state.name == "FAILED":
            raise ValueError("Video processing failed on Google's side.")
        elif uploaded_file.state.name != "ACTIVE":
             raise ValueError(f"Unexpected video file state: {uploaded_file.state.name}")

        status_placeholder.info(f"🧠 Analyzing {skill_name} with Gemini...")

        # 3. Make the Gemini API call
        response = model.generate_content([prompt, uploaded_file], request_options={"timeout": 120}) # 2 min timeout for generation

        # 4. Parse the response
        try:
            # Check for safety blocks or empty response
            if not response.parts:
                 st.warning(f"⚠️ Gemini response empty or blocked for {skill_name}. Score set to 0.")
                 score = 0
            else:
                raw_score_text = response.text.strip()
                score = int(raw_score_text)
                # Clamp the score between 0 and MAX_SCORE_PER_SKILL
                score = max(0, min(MAX_SCORE_PER_SKILL, score))
                status_placeholder.success(f"✅ Analysis complete for {skill_name}. Raw score text: '{raw_score_text}', Parsed score: {score}")
        except ValueError:
            st.warning(f"⚠️ Gemini response for {skill_name} was not a valid number ('{response.text}'). Score set to 0.")
            score = 0
        except Exception as e:
            st.warning(f"⚠️ Error parsing Gemini response for {skill_name}: {e}. Score set to 0.")
            score = 0

    except genai.types.generation_types.BlockedPromptException as bpe:
         st.error(f"❌ Prompt blocked by safety settings for {skill_name}: {bpe}")
         score = 0
    except genai.types.generation_types.StopCandidateException as sce:
         st.error(f"❌ Analysis stopped unexpectedly for {skill_name} (Safety/Policy): {sce}")
         score = 0
    except TimeoutError as te:
         st.error(f"❌ Timeout error during analysis for {skill_name}: {te}")
         score = 0
    except Exception as e:
        st.error(f"❌ An error occurred during Gemini analysis for {skill_name}: {e}")
        score = 0
    finally:
        # 5. Clean up the uploaded file on Google Cloud
        if uploaded_file:
            try:
                status_placeholder.info(f"🗑️ Deleting uploaded file for {skill_name} from cloud storage...")
                genai.delete_file(uploaded_file.name)
                status_placeholder.info(f"🗑️ Cloud file deleted for {skill_name}.")
            except Exception as e:
                st.warning(f"⚠️ Could not delete uploaded file {uploaded_file.name}: {e}")

    return score


# =========== Grading and Plotting Functions (same as before) =================

# --- Final Grade Calculation ---
def evaluate_final_grade_from_individual_scores(scores_dict):
    total = sum(scores_dict.values())
    max_possible = len(scores_dict) * MAX_SCORE_PER_SKILL
    percentage = (total / max_possible) * 100 if max_possible > 0 else 0

    if percentage >= 90: grade = 'ممتاز (A)' # Excellent
    elif percentage >= 75: grade = 'جيد جداً (B)' # Very Good
    elif percentage >= 55: grade = 'جيد (C)'    # Good
    elif percentage >= 40: grade = 'مقبول (D)'  # Acceptable
    else: grade = 'ضعيف (F)'                   # Weak / Needs Improvement

    return {"scores": scores_dict, "total_score": total, "grade": grade, "max_score": max_possible}

# --- Plotting Function ---
def plot_results(results):
    skills_arabic = {
        'Jumping': 'القفز', 'Running': 'الجري', 'Passing': 'التمرير',
        'Receiving': 'الاستقبال', 'Zigzag': 'المراوغة'
    }
    try:
        labels = [get_display(arabic_reshaper.reshape(skills_arabic.get(en, en))) for en in results['scores'].keys()]
        plot_title = get_display(arabic_reshaper.reshape(f"التقييم النهائي - التقدير: {results['grade']} ({results['total_score']}/{results['max_score']})"))
        y_axis_label = get_display(arabic_reshaper.reshape(f"الدرجة (من {MAX_SCORE_PER_SKILL})"))
    except Exception as e:
        st.warning(f"Could not reshape Arabic text for plot: {e}")
        labels = list(results['scores'].keys())
        plot_title = f"Final Evaluation - Grade: {results['grade']} ({results['total_score']}/{results['max_score']})"
        y_axis_label = f"Score (out of {MAX_SCORE_PER_SKILL})"

    scores = list(results['scores'].values())
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, scores)
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

# =========== Streamlit App Layout ===================================

# Initialize session state
if 'page' not in st.session_state: st.session_state.page = 'home'
if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = None
if 'uploaded_files_dict' not in st.session_state: st.session_state.uploaded_files_dict = {}

# --- Top Row: Logo ---
col1, col_mid, col2 = st.columns([1, 3, 1])
with col1:
    st.markdown("<p style='font-size: 1.2em; font-weight: bold;'>AI LEAGUE</p>", unsafe_allow_html=True) # Placeholder text

# --- Center Area: Main Logo, Title, Slogan ---
st.container()
st.markdown("<h1 style='text-align: center; color: white; margin-top: 20px;'>Scout Eye</h1>", unsafe_allow_html=True)
st.markdown("<div class='title-text'>عين الكشاف</div>", unsafe_allow_html=True)
st.markdown("<div class='slogan-text'>نكتشف ، نحمي ، ندعم</div>", unsafe_allow_html=True)

# --- Bottom Area: Clickable Options ---
st.container()
col_b1, col_b2, col_b3 = st.columns(3)
button_keys = ["btn_person", "btn_star", "btn_legend"]

if col_b1.button("الشخص المناسب ✔️", key=button_keys[0]):
    st.session_state.page = 'الشخص_المناسب'
    st.session_state.evaluation_results = None
if col_b2.button("نجم لا يغيب ⭐", key=button_keys[1]):
    st.session_state.page = 'نجم_لا_يغيب'
    st.session_state.evaluation_results = None
if col_b3.button("إسطورة الغد ⚽", key=button_keys[2]):
    st.session_state.page = 'اسطورة_الغد'
    st.session_state.evaluation_results = None

# --- Conditional Page Content ---

# ==================================
# ==      إسطورة الغد Page       ==
# ==================================
if st.session_state.page == 'اسطورة_الغد':
    st.markdown("---")
    st.markdown("## ⚽ إسطورة الغد - التقييم الشامل (بواسطة Gemini) ⚽")
    st.markdown("<p style='text-align: center; font-size: 1.1em;'>قم برفع مقاطع فيديو منفصلة لكل مهارة ليتم تحليلها بواسطة الذكاء الاصطناعي Gemini.</p>", unsafe_allow_html=True)
    st.warning("⚠️ ملاحظة: تحليل الفيديو باستخدام Gemini API قد يستغرق بعض الوقت ويتطلب تكلفة API.")

    uploaded_files_local = {}
    col_upload1, col_upload2 = st.columns(2)

    with col_upload1:
        st.markdown("### المهارات الأساسية")
        for skill, label in list(SKILLS_TO_EVALUATE.items())[:3]:
            uploaded_files_local[skill] = st.file_uploader(label, type=["mp4", "avi", "mov", "mkv", "webm"], key=f"upload_{skill}")
    with col_upload2:
        st.markdown("### مهارات التحكم والمراوغة")
        for skill, label in list(SKILLS_TO_EVALUATE.items())[3:]:
             uploaded_files_local[skill] = st.file_uploader(label, type=["mp4", "avi", "mov", "mkv", "webm"], key=f"upload_{skill}")

    # Store uploaded files in session state immediately to persist them
    st.session_state.uploaded_files_dict = uploaded_files_local

    st.markdown("---")

    if st.button("🚀 بدء التقييم الكامل بواسطة Gemini", key="start_gemini_eval"):
        # Check if all files are uploaded from session state
        all_files_present = all(st.session_state.uploaded_files_dict.get(skill) for skill in SKILLS_TO_EVALUATE)

        if all_files_present:
            st.info("⏳ Iniciando análisis de video con Gemini para todas las habilidades...")
            results_dict = {}
            error_occurred = False
            local_temp_files = {} # To store paths of local temp files

            with st.status("📊 تحليل المهارات باستخدام Gemini...", expanded=True) as status_main:
                try:
                    # Process each skill
                    for skill, uploaded_file_obj in st.session_state.uploaded_files_dict.items():
                        if uploaded_file_obj:
                            status_skill = st.empty() # Placeholder for individual skill status
                            status_skill.write(f"Processing {skill}...")

                            # Save video bytes to a temporary local file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_obj.name)[1]) as tmp_file:
                                tmp_file.write(uploaded_file_obj.getvalue())
                                video_path = tmp_file.name
                                local_temp_files[skill] = video_path # Store path for later deletion

                            # Call Gemini analysis function
                            score = get_skill_score_from_gemini(video_path, skill, status_skill)
                            results_dict[skill] = score
                            if score == 0: # Consider score 0 as a potential issue (though it might be valid)
                                st.warning(f"Score for {skill} is 0. Check video content or Gemini analysis.")
                        else:
                            st.error(f"File for {skill} is missing!")
                            error_occurred = True
                            break # Stop if a file is unexpectedly missing

                    if not error_occurred:
                        status_main.update(label="✅ اكتمل تحليل Gemini لجميع المهارات!", state="complete", expanded=False)
                        st.session_state.evaluation_results = evaluate_final_grade_from_individual_scores(results_dict)
                        st.success("🎉 تم حساب التقييم النهائي بنجاح!")
                        st.balloons()
                    else:
                         status_main.update(label="❌ فشل تحليل بعض المهارات.", state="error", expanded=True)
                         st.session_state.evaluation_results = None

                except Exception as e:
                    st.error(f"❌ حدث خطأ فادح أثناء عملية التحليل الشاملة: {e}")
                    error_occurred = True
                    status_main.update(label="❌ فشل التحليل", state="error")
                    st.session_state.evaluation_results = None
                finally:
                    # Clean up local temporary files
                    for skill, path in local_temp_files.items():
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                                # st.write(f"Deleted local temp file for {skill}") # Debug
                            except Exception as e_del:
                                st.warning(f"Could not delete local temp file {path}: {e_del}")


        else:
            st.error("✋ الرجاء رفع جميع مقاطع الفيديو الخمسة المطلوبة لبدء التقييم.")
            st.session_state.evaluation_results = None

    # --- Display Stored Results ---
    if st.session_state.evaluation_results:
        results = st.session_state.evaluation_results
        st.markdown("---")
        st.markdown("### 🏆 النتائج النهائية (بناءً على تحليل Gemini) 🏆")
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
             st.write(results['scores'])

# ==================================
# ==    Other Pages (Placeholders) ==
# ==================================
elif st.session_state.page == 'نجم_لا_يغيب':
    st.markdown("---")
    st.markdown("## ⭐ نجم لا يغيب ⭐")
    st.info("هذه الميزة ستستخدم Gemini API لتحليل جوانب أخرى من أداء اللاعب.")
    # TODO: Add UI and logic for this page using Gemini

elif st.session_state.page == 'الشخص_المناسب':
    st.markdown("---")
    st.markdown("## ✔️ الشخص المناسب في المكان المناسب ✔️")
    st.info("هذه الميزة ستستخدم Gemini API لتحليل مجموعة بيانات (سيتم تحديدها لاحقاً).")
    # TODO: Add UI and logic for this page using Gemini


# --- Footer ---
st.markdown("---")
st.caption("AI League - Scout Eye v1.0 (Gemini Powered) | Powered by Google Gemini API")