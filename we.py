import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎓 Gợi ý môn học và tài liệu học tập")

# Load dữ liệu phản hồi sinh viên
@st.cache_data
def load_data():
    df = pd.read_csv("course_reviews_5k_final.csv")
    df["Nội dung phản hồi chuẩn hóa"] = df["Nội dung phản hồi"].str.lower().str.replace(r"[^a-zA-ZÀ-ỹ\s]", "", regex=True)
    return df

df = load_data()

# Gộp văn bản phản hồi theo môn học
merged_texts_by_course = df.groupby("Tên môn học")["Nội dung phản hồi chuẩn hóa"].apply(lambda texts: " ".join(texts)).to_dict()
course_list = list(merged_texts_by_course.keys())
course_documents = [merged_texts_by_course[c] for c in course_list]

# TF-IDF và Cosine Similarity
tfidf = TfidfVectorizer()
course_vectors = tfidf.fit_transform(course_documents)
similarity_df = pd.DataFrame(cosine_similarity(course_vectors), index=course_list, columns=course_list)

# Độ khó trung bình chính xác
difficulty_scores = df.groupby("Tên môn học").apply(
    lambda g: round((g["Số sao"] * g["Tổng lượt đánh giá"]).sum() / g["Tổng lượt đánh giá"].sum(), 1)
).to_dict()

# Tài liệu học tập theo chuyên ngành
resources = {
    "AI": ["Deep Learning Specialization - Coursera", "FastAI Practical Course", "AI for Everyone - Andrew Ng"],
    "ML": ["Machine Learning - Stanford", "Hands-On ML with Scikit-Learn", "ML Crash Course - Google"],
    "Signals": ["Signals & Systems - MIT", "Digital Signal Processing - Coursera"],
    "Networks": ["Computer Networking - Stanford", "Cisco CCNA"],
    "Embedded": ["Embedded Systems - Coursera", "Arduino for Beginners"],
    "Programming": ["Python for Everybody", "C++ for Programmers"],
    "Database": ["Intro to SQL - Khan Academy", "Database Design - Coursera"],
    "Math": ["Calculus - Khan Academy", "Linear Algebra - MIT"]
}

def get_resources(course_name):
    name = course_name.lower()
    if "ai" in name:
        return resources["AI"]
    elif "machine learning" in name or "học máy" in name:
        return resources["ML"]
    elif "tín hiệu" in name:
        return resources["Signals"]
    elif "mạng" in name:
        return resources["Networks"]
    elif "nhúng" in name:
        return resources["Embedded"]
    elif "lập trình" in name or "python" in name or "c++" in name:
        return resources["Programming"]
    elif "cơ sở dữ liệu" in name:
        return resources["Database"]
    elif "toán" in name or "xác suất" in name:
        return resources["Math"]
    else:
        return ["Chưa có gợi ý cụ thể"]

def suggest_related(course_name, top_n=3):
    if course_name not in similarity_df:
        return []
    sim_scores = similarity_df[course_name].sort_values(ascending=False)[1:top_n+1]
    return [(idx, round(score, 2)) for idx, score in sim_scores.items()]

# Giao diện chọn môn học
selected_course = st.selectbox("Chọn môn học bạn muốn tìm hiểu", sorted(course_list))

if selected_course:
    st.markdown("---")
    st.subheader(f"📊 Độ khó trung bình của {selected_course}:")
    st.write(f"{difficulty_scores.get(selected_course, 'Không có dữ liệu')} ⭐")

    st.subheader("🔗 Môn học liên quan")
    for related, score in suggest_related(selected_course):
        st.write(f"- {related} (similarity: {score})")

    st.subheader("📚 Tài liệu học tập gợi ý")
    for r in get_resources(selected_course):
        st.write("-", r)
