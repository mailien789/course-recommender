import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Bước 1: Đọc file đã gắn từ khóa ---
df = pd.read_csv("course_reviews_5k_final.csv")

# --- Bước 2: Chuẩn hóa nội dung phản hồi ---
df["Nội dung phản hồi chuẩn hóa"] = df["Nội dung phản hồi"].str.lower().str.replace(r"[^a-zA-ZÀ-ỹ\s]", "", regex=True)

# --- Bước 3: Gộp nội dung phản hồi theo môn học ---
merged_texts_by_course = df.groupby("Tên môn học")["Nội dung phản hồi chuẩn hóa"].apply(lambda texts: " ".join(texts)).to_dict()
course_list = list(merged_texts_by_course.keys())
course_documents = [merged_texts_by_course[c] for c in course_list]

# --- Bước 4: TF-IDF vectorization ---
tfidf = TfidfVectorizer()
course_vectors = tfidf.fit_transform(course_documents)

# --- Bước 5: Cosine Similarity ---
cos_sim_matrix = cosine_similarity(course_vectors)

# --- Bước 6: Tìm Top 3 môn học liên quan ---
related_courses = {}
for i, course in enumerate(course_list):
    sims = list(enumerate(cos_sim_matrix[i]))
    sims = sorted([s for s in sims if s[0] != i and s[1] > 0.5], key=lambda x: x[1], reverse=True)[:3]
    related_courses[course] = [(course_list[j], round(score, 2)) for j, score in sims]

# --- Bước 7: Tính độ khó trung bình đúng ---
difficulty_scores = df.groupby("Tên môn học").apply(
    lambda g: round((g["Số sao"] * g["Tổng lượt đánh giá"]).sum() / g["Tổng lượt đánh giá"].sum(), 1)
).to_dict()

# --- Bước 8: Gợi ý khóa học online liên quan ---
related_courses_resources = {
    "AI": ["Deep Learning Specialization - Coursera", "AI for Everyone - Andrew Ng", "FastAI Practical Course", "Intro to AI - edX"],
    "ML": ["Machine Learning - Stanford", "Intro to ML with PyTorch - Udacity", "Hands-On ML with Scikit-Learn", "ML Crash Course - Google"],
    "Data": ["SQL for Data Science", "Data Analysis with Pandas", "Big Data with Hadoop", "Data Science Specialization - Coursera"],
    "Embedded": ["Embedded Systems - Coursera", "Arduino for Beginners", "Real-time Embedded Systems", "Microcontroller Programming - Udemy"],
    "Networks": ["Computer Networking - Stanford", "Cisco CCNA", "Networking Fundamentals - Udemy", "Practical Network Protocols"],
    "Signals": ["Digital Signal Processing - Coursera", "Signals & Systems - MIT", "Practical DSP", "DSP with MATLAB"],
    "Database": ["Intro to SQL - Khan Academy", "Database Design - Coursera", "MongoDB University", "Advanced SQL Techniques"],
    "Programming": ["C++ for Programmers", "Python for Everybody", "Java Programming Masterclass", "Object-Oriented Programming - Udemy"],
    "Math": ["Calculus - Khan Academy", "Linear Algebra - MIT OCW", "Probability & Statistics - Udacity", "Discrete Mathematics - Coursera"]
}

def suggest_related_courses(course_name):
    name = course_name.lower()
    if "ai" in name:
        return related_courses_resources["AI"]
    elif "máy học" in name or "learning" in name:
        return related_courses_resources["ML"]
    elif "dữ liệu" in name or "sql" in name:
        return related_courses_resources["Database"]
    elif "nhúng" in name:
        return related_courses_resources["Embedded"]
    elif "mạng" in name:
        return related_courses_resources["Networks"]
    elif "tín hiệu" in name:
        return related_courses_resources["Signals"]
    elif "toán" in name or "xác suất" in name:
        return related_courses_resources["Math"]
    elif "lập trình" in name or "python" in name or "c++" in name:
        return related_courses_resources["Programming"]
    else:
        return ["No specific course recommendations found."]

# --- Bước 9: Tạo kết quả tổng hợp ---
results = []
for course in course_list:
    related = related_courses.get(course, [])
    resources = suggest_related_courses(course)
    difficulty = difficulty_scores.get(course, 0)
    results.append({
        "Môn học": course,
        "Độ khó trung bình": difficulty,
        "Môn liên quan (Top 3)": [f"{c} (similarity: {s})" for c, s in related],
        "Khóa học gợi ý": resources
    })

# --- Bước 10: Lưu kết quả ra file ---
results_df = pd.DataFrame(results)
results_df.to_csv("course_related_with_resources.csv", index=False, encoding="utf-8-sig")
