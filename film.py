import pandas as pd
import csv

# Tạo dữ liệu phim mẫu tiếng Việt
movies_data = [
    {
        "title": "Phim Test 1",
        "genre": "Hành Động|Viễn Tưởng",
        "film_type": "Phim Lẻ",
        "director": "Đạo diễn A",
        "actors": "Diễn viên X, Diễn viên Y",
        "plot": "Nội dung phim test 1",
        "year": 2021,
        "rating": 8.5,
        "country": "Việt Nam"
    },
    {
        "title": "Phim Test 2",
        "genre": "Tình Cảm|Hài Hước",
        "film_type": "Phim Bộ",
        "director": "Đạo diễn B",
        "actors": "Diễn viên Z, Diễn viên W",
        "plot": "Nội dung phim test 2",
        "year": 2022,
        "rating": 7.8,
        "country": "Hàn Quốc"
    },
    {
        "title": "Phim Test 3",
        "genre": "Kinh Dị|Hình Sự",
        "film_type": "Phim Lẻ",
        "director": "Đạo diễn C",
        "actors": "Diễn viên M, Diễn viên N",
        "plot": "Nội dung phim test 3",
        "year": 2023,
        "rating": 8.2,
        "country": "Mỹ"
    },
    {
        "title": "Phim Test 4",
        "genre": "Hoạt Hình|Gia Đình",
        "film_type": "Phim Hoạt Hình",
        "director": "Đạo diễn D",
        "actors": "Lồng tiếng P, Lồng tiếng Q",
        "plot": "Nội dung phim test 4",
        "year": 2020,
        "rating": 9.0,
        "country": "Nhật Bản"
    },
    {
        "title": "Phim Test 5",
        "genre": "Cổ Trang|Võ Thuật",
        "film_type": "Phim Bộ",
        "director": "Đạo diễn E",
        "actors": "Diễn viên R, Diễn viên S",
        "plot": "Nội dung phim test 5",
        "year": 2019,
        "rating": 8.7,
        "country": "Trung Quốc"
    }
]

# Tạo DataFrame từ dữ liệu
movies_df = pd.DataFrame(movies_data)

# Lưu vào file CSV
movies_df.to_csv('vietnamese_movies.csv', index=False)
print("Đã tạo file vietnamese_movies.csv với 5 bộ phim mẫu.")
