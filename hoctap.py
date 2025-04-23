import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv("course_reviews_real_5k.csv", encoding="utf-8-sig")


# 2. Gộp nội dung phản hồi theo từng môn học
grouped = df.groupby("Tên môn học")["Nội dung phản hồi"].apply(lambda texts: " ".join(texts)).reset_index()


# 3. Biến văn bản thành vector bằng TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(grouped["Nội dung phản hồi"])


# 4. Tính ma trận Cosine Similarity giữa các môn
cosine_sim = cosine_similarity(tfidf_matrix)


# 5. Đưa vào DataFrame để dễ xem
similarity_df = pd.DataFrame(cosine_sim, index=grouped["Tên môn học"], columns=grouped["Tên môn học"])


# 6. Ví dụ: Xem top 5 môn học liên quan đến "Hệ điều hành"
course_name = "Hệ điều hành"
top_related = similarity_df[course_name].sort_values(ascending=False)[1:6]
print(f"\nTop môn học liên quan đến '{course_name}':\n")
print(top_related)

def suggest_related_courses(course_name, top_n=5):
    if course_name not in similarity_df:
        return "Không tìm thấy môn học!"
   
    related = similarity_df[course_name].sort_values(ascending=False)[1:top_n+1]
    return related

def suggest_top_lecturers(course_name, min_rating=4.0):
    df_course = df[df["Tên môn học"] == course_name]
    df_course = df_course.groupby("Họ tên").agg({"Số sao": "mean", "Mã số sinh viên": "count"}).reset_index()
    df_course = df_course[df_course["Số sao"] >= min_rating]
    df_course = df_course.sort_values(by="Số sao", ascending=False)
    return df_course.head(3)

course_resources = {
    "Hệ điều hành": [
        "Sách: Operating System Concepts – Silberschatz",
        "Video: CPU Scheduling Explained (YouTube)",
        "Khóa học: CS50 Operating Systems (Harvard)"
    ],
}


def suggest_resources(course_name):
    return course_resources.get(course_name, ["Chưa có tài liệu gợi ý"])