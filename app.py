
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Đọc dữ liệu
df = pd.read_csv("course_reviews_real_5k.csv", encoding="utf-8-sig")

# Gộp phản hồi theo môn học
grouped = df.groupby("Tên môn học")["Nội dung phản hồi"].apply(lambda texts: " ".join(texts)).reset_index()

# TF-IDF và Cosine Similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(grouped["Nội dung phản hồi"])
cosine_sim = cosine_similarity(tfidf_matrix)
similarity_df = pd.DataFrame(cosine_sim, index=grouped["Tên môn học"], columns=grouped["Tên môn học"])

# Gợi ý môn liên quan
def suggest_related_courses(course_name, top_n=5):
    if course_name not in similarity_df:
        return "Không tìm thấy môn học!"
    related = similarity_df[course_name].sort_values(ascending=False)[1:top_n+1]
    return related

# Gợi ý giảng viên chất lượng
def suggest_top_lecturers(course_name, min_rating=4.0):
    df_course = df[df["Tên môn học"] == course_name]
    df_course = df_course.groupby("Họ tên").agg({"Số sao": "mean", "Mã số sinh viên": "count"}).reset_index()
    df_course = df_course[df_course["Số sao"] >= min_rating]
    df_course = df_course.sort_values(by="Số sao", ascending=False)
    return df_course.head(3)

# Gợi ý tài liệu
course_resources = {
    "Hệ điều hành": [
        "Sách: Operating System Concepts – Silberschatz",
        "Video: CPU Scheduling Explained (YouTube)",
        "Khóa học: CS50 Operating Systems (Harvard)"
    ],
    "Cấu trúc dữ liệu": [
        "Sách: Data Structures and Algorithms in Python",
        "Video: Linked Lists Explained (YouTube)"
    ],
    "Lập trình Java": [
        "Sách: Effective Java – Joshua Bloch",
        "Video: Java OOP Full Course"
    ]
}

def suggest_resources(course_name):
    return course_resources.get(course_name, ["Chưa có tài liệu gợi ý"])

# Giao diện
st.title("🎓 Hệ thống gợi ý môn học")

course_input = st.text_input("Nhập tên môn học bạn muốn tìm hiểu")

if st.button("Gợi ý"):
    if course_input:
        st.subheader("🔗 Môn học liên quan:")
        st.write(suggest_related_courses(course_input))

        st.subheader("👨‍🏫 Giảng viên được đánh giá cao:")
        st.write(suggest_top_lecturers(course_input))

        st.subheader("📚 Tài liệu học tập:")
        for res in suggest_resources(course_input):
            st.write("-", res)
    else:
        st.warning("Vui lòng nhập tên môn học.")

