import pandas as pd
import random

# Đọc file danh sách môn học đã xử lý
courses_df = pd.read_csv("/mnt/data/courses_data.csv")

# Từ điển mô tả môn học đầy đủ (trích từ file PDF)
course_descriptions = {
    "Toán 1": "Giới thiệu các kiến thức về giới hạn, đạo hàm, tích phân, chuỗi số và hàm.",
    "Toán 2": "Ma trận, hệ phương trình tuyến tính, không gian vector, chéo hóa ma trận và đạo hàm hàm nhiều biến.",
    "Toán 3": "Tích phân bội, tích phân đường, tích phân mặt và ứng dụng trong hình học không gian.",
    "Xác suất thống kê ứng dụng": "Lý thuyết xác suất, phân phối xác suất, kiểm định giả thuyết và hồi quy.",
    "Vật lý 1": "Cơ học, nhiệt động lực học và điện từ học cơ bản.",
    "Vật lý 2": "Thuyết tương đối, quang học và vật lý lượng tử.",
    "Ngôn ngữ lập trình C": "Lập trình cơ bản với ngôn ngữ C, cấu trúc điều khiển, mảng, con trỏ và hàm.",
    "Nhập môn ngành CNKT máy tính": "Tổng quan ngành, kỹ năng mềm và phương pháp học tập hiệu quả.",
    "Mạch điện": "Định luật Kirchhoff, các phương pháp phân tích mạch, mạch ba pha, mạng hai cửa và phân tích miền tần số.",
    "Điện tử cơ bản": "Linh kiện bán dẫn, mạch chỉnh lưu, khuếch đại và mạch dao động.",
    "Kỹ thuật số": "Hệ thống số, đại số Boole, vi mạch số và thiết kế hệ thống số.",
    "Tín hiệu và hệ thống": "Phân tích tín hiệu, biến đổi Laplace, tích chập, phân tích tần số và hệ thống lọc.",
    "Kỹ thuật truyền số liệu": "Dồn kênh, điều khiển lỗi, dịch vụ dữ liệu và truyền số liệu qua mạng.",
    "Cấu trúc rời rạc": "Tập hợp, quan hệ, đồ thị, cây và ngôn ngữ hình thức.",
    "Kiến trúc và tổ chức máy tính": "Vi kiến trúc, tổ chức bộ nhớ, thiết bị ngoại vi và lập trình hợp ngữ.",
    "Hệ thống nhúng": "Thiết kế và lập trình hệ thống nhúng với Arduino và các nền tảng nhúng.",
    "Thiết kế FPGA/ASIC với Verilog": "Thiết kế mạch tổ hợp, tuần tự với Verilog cho FPGA và ASIC.",
    "Xử lý tín hiệu số": "Lấy mẫu tín hiệu, biến đổi Z, DTFT, DFT và thiết kế bộ lọc số.",
    "Mạng máy tính và Internet": "Các giao thức TCP/IP, thiết bị mạng và thiết kế hệ thống mạng.",
    "Hệ điều hành thời gian thực": "Tiến trình, đồng bộ hóa, lập lịch và thiết kế hệ điều hành nhúng thời gian thực.",
    "Thiết kế kết hợp HW/SW": "Thiết kế phần cứng - phần mềm tích hợp cho hệ thống nhúng.",
    "Thiết kế mạch tích hợp VLSI": "Thiết kế cổng logic tổ hợp, tuần tự và hệ thống vi mạch số.",
    "Cơ sở và ứng dụng IoT": "Nền tảng phần cứng, giao thức M2M và xử lý dữ liệu trong IoT.",
    "Lập trình hướng đối tượng với C++": "Lập trình C++: lớp, đối tượng, kế thừa, đa hình và thiết kế OOP.",
    "Giải thuật và cấu trúc dữ liệu": "Cấu trúc danh sách, cây, bảng băm và các giải thuật tìm kiếm, sắp xếp.",
    "Thiết kế vi mạch tương tự": "Thiết kế mạch khuếch đại, bộ dòng, DRAM, SRAM và Flash.",
    "Máy học ứng dụng": "Nhận diện mẫu, học giám sát, không giám sát và hệ thống khuyến nghị.",
    "Mạng vô tuyến và di động": "Mạng không dây, GSM, LTE và truyền dữ liệu di động.",
    "Phát triển ứng dụng di động": "Lập trình ứng dụng Android, xây dựng và triển khai ứng dụng thực tế.",
    "Cơ sở và ứng dụng AI": "Toán ứng dụng, mạng nơ-ron, học sâu và lập trình AI với Python.",
    "Điện toán đám mây": "Xử lý song song, lưu trữ phân tán, bảo mật và lập trình trên đám mây.",
    "Hệ cơ sở dữ liệu": "Thiết kế cơ sở dữ liệu, truy vấn SQL, lập chỉ mục và xử lý giao dịch.",
    "Thiết kế hệ thống nhúng": "Thiết kế và kiểm lỗi ứng dụng phần mềm cho hệ thống nhúng phức tạp.",
    "Khóa luận tốt nghiệp": "Nghiên cứu và phát triển đồ án chuyên ngành kỹ thuật máy tính."
}

# Danh sách các từ khóa kỹ thuật để random vào nội dung phản hồi
topics = [
    "xử lý tín hiệu", "lập trình nhúng", "AI", "IoT", "thiết kế vi mạch",
    "mạng máy tính", "điện toán đám mây", "cấu trúc dữ liệu", "giải thuật tối ưu",
    "kỹ thuật số", "thực hành phần cứng"
]

# Hàm sinh phản hồi theo mô tả
def generate_custom_review(course_name):
    description = course_descriptions.get(course_name, "Môn học cung cấp kiến thức nền tảng về kỹ thuật máy tính.")
    keywords = random.sample(topics, k=3)
    return f"{description} Môn học giúp sinh viên tiếp cận các chủ đề như {', '.join(keywords)}."

# Thêm cột nội dung phản hồi vào DataFrame
courses_df["Nội dung phản hồi"] = courses_df["Tên môn học"].apply(generate_custom_review)

# Xuất file CSV mới
updated_path = "/mnt/data/course_reviews_customized.csv"
courses_df.to_csv(updated_path, index=False, encoding="utf-8-sig")

# Trả về đường dẫn file mới
updated_path
