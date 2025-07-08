import streamlit as st 
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
career_encoder = joblib.load("career_label_encoder.pkl")
model_input_columns = joblib.load("model_input_columns.pkl")
categorical_cols = ["Stream", "Activity", "Interest1", "Interest2", "Skill1", "Skill2", "Subject1", "Subject2", "PreferredEnv", "StudyStyle"]
encoders = {col: joblib.load(f"{col.lower()}_encoder.pkl") for col in categorical_cols}

# Shared input storage
if "inputs" not in st.session_state:
    st.session_state.inputs = {}

st.title("AI-Driven Career Recommendation System")

# Step 1: Profile Information
st.subheader("Step 1: Profile Information")
for col in categorical_cols:
    st.session_state.inputs[col] = st.selectbox(f"Select {col}", encoders[col].classes_, key=col)

# Step 2: Numerical Ability Quiz
st.subheader("Step 2: Numerical Ability (10 Questions)")
numerical_questions = [
    ("742.25 × 2.5 + 128.5 – 1489.88 = ?", ["4291.14", "4301.14", "5941.14", "4201.14"], "4301.14"),
    ("8054 + 6911 + 3847 = ?", ["18722", "18992", "18812", "18092"], "18812"),
    ("100 ÷ 4 + 6 = ?", ["31", "28", "34", "25"], "31"),
    ("What is 15% of 200?", ["30", "25", "45", "20"], "30"),
    ("The square of 12 is?", ["144", "124", "132", "128"], "144"),
    ("A man earns ₹5000 and spends ₹3200. Savings?", ["₹1800", "₹1500", "₹2000", "₹2200"], "₹1800"),
    ("Which is larger: 0.7 or 0.77?", ["0.7", "0.77"], "0.77"),
    ("What is 13²?", ["169", "179", "189", "159"], "169"),
    ("Convert 0.5 into percentage", ["50%", "5%", "0.5%", "25%"], "50%"),
    ("What is 10 × 12 ÷ 4?", ["30", "25", "28", "35"], "30")
]
n_score = 0
for i, (q, opts, ans) in enumerate(numerical_questions):
    user_ans = st.radio(f"Q{i+1}: {q}", opts, index=None, key=f"num_{i}")
    if user_ans == ans:
        n_score += 1
st.session_state.inputs["Numerical"] = n_score

# Step 3: Logical Reasoning
st.subheader("Step 3: Logical Reasoning (10 Questions)")
logical_questions = [
    ("Which number comes next: 2, 4, 8, 16, ...?", ["18", "32", "24", "30"], "32"),
    ("If 2 pencils cost ₹8, how many can you buy with ₹40?", ["5", "8", "10", "12"], "10"),
    ("Find the odd one: Apple, Orange, Banana, Carrot", ["Apple", "Orange", "Banana", "Carrot"], "Carrot"),
    ("Which one is different: 12, 18, 20, 24", ["12", "18", "20", "24"], "20"),
    ("If A=1, B=2, then Z=?", ["26", "24", "25", "27"], "26"),
    ("Which comes next: Monday, Tuesday, Wednesday, ...?", ["Thursday", "Friday", "Saturday"], "Thursday"),
    ("5 birds on a tree, 2 fly away. Birds left?", ["3", "2", "5", "0"], "3"),
    ("Find missing: 3, 6, 12, 24, ?", ["36", "48", "40", "50"], "48"),
    ("Which shape has no sides?", ["Square", "Circle", "Triangle", "Hexagon"], "Circle"),
    ("Which is the mirror image of 'b'?", ["d", "p", "q", "b"], "d")
]
l_score = 0
for i, (q, opts, ans) in enumerate(logical_questions):
    user_ans = st.radio(f"Q{i+1}: {q}", opts, index=None, key=f"log_{i}")
    if user_ans == ans:
        l_score += 1
st.session_state.inputs["Logical"] = l_score

# Step 4: English Grammar
st.subheader("Step 4: English Grammar (10 Questions)")
english_questions = [
    ("Choose the correct sentence:", ["He go to school.", "He goes to school.", "He going to school."], "He goes to school."),
    ("Opposite of 'generous':", ["Kind", "Selfish", "Honest"], "Selfish"),
    ("Plural of 'child' is:", ["Childs", "Children", "Childes"], "Children"),
    ("Which word is a noun?", ["Quickly", "Happiness", "Blue", "Run"], "Happiness"),
    ("Choose correct spelling:", ["Accomodate", "Acommodate", "Accommodate"], "Accommodate"),
    ("Past tense of 'eat' is:", ["Eated", "Ate", "Eats"], "Ate"),
    ("Choose the synonym of 'begin':", ["End", "Start", "Stop", "Pause"], "Start"),
    ("Which one is an adjective?", ["Big", "Run", "Blue", "Quickly"], "Big"),
    ("Fill the blank: He ____ reading a book.", ["is", "are", "were", "was"], "is"),
    ("Identify the verb: The dog barks loudly.", ["Dog", "Barks", "Loudly", "The"], "Barks")
]
e_score = 0
for i, (q, opts, ans) in enumerate(english_questions):
    user_ans = st.radio(f"Q{i+1}: {q}", opts, index=None, key=f"eng_{i}")
    if user_ans == ans:
        e_score += 1
st.session_state.inputs["English"] = e_score

# Step 5: Personality Assessment
st.subheader("Step 5: Personality Assessment (10 Questions)")
personality_questions = [
    "I enjoy helping others even if I don’t benefit.",
    "I prefer working alone rather than in a team.",
    "I like to take charge of group activities.",
    "I remain calm under pressure.",
    "I enjoy solving problems creatively.",
    "I adapt quickly to new situations.",
    "I stay focused on my goals.",
    "I like to organize tasks and schedules.",
    "I enjoy speaking in front of others.",
    "I stay positive even when things go wrong."
]
p_score = 0
for i, q in enumerate(personality_questions):
    p_score += st.slider(f"{i+1}. {q}", 1, 5, 1, key=f"personality_{i}")
st.session_state.inputs["Personality"] = p_score

# Prediction Button and Output
st.markdown("---")
st.header("Recommended Careers")
if st.button("Predict My Career"):
    try:
        encoded_input = {}
        for col in categorical_cols:
            encoded_input[col] = encoders[col].transform([st.session_state.inputs[col]])[0]

        for score_col in ["Numerical", "Logical", "English", "Personality"]:
            encoded_input[score_col] = st.session_state.inputs.get(score_col, 0)

        input_df = pd.DataFrame([encoded_input])
        for col in model_input_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_input_columns]

        # Removed the line that displays model input
        # st.write("Model Input Data:", input_df)

        proba = model.predict_proba(input_df)[0]
        top3 = sorted(enumerate(proba), key=lambda x: -x[1])[:3]
        for idx, prob in top3:
            career = career_encoder.inverse_transform([idx])[0]
            st.success(f"{career}: {prob * 100:.2f}% confidence")
    except Exception as e:
        st.error(f"Prediction error: {e}")
