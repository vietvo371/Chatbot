import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from flask import Flask, request, jsonify,session
import secrets
from flask_cors import CORS
import os
import re
import datetime
import random
import json
import logging
import mysql.connector
from mysql.connector import pooling  # Thêm dòng import này
from dotenv import load_dotenv
from functools import lru_cache
import time
import json
import uuid
from typing import Dict, List, Optional, Union, Any

# Tải biến môi trường từ file .env
load_dotenv()

# Thiết lập logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('movie_chatbot')

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Định nghĩa danh sách thể loại phim tiếng Việt
VIETNAMESE_GENRES = [
    "Viễn Tưởng", "Tình Cảm", "Tài Liệu",
    "Khoa Học", "Chiến Tranh", "Âm Nhạc",
    "Chính Kịch", "Gia Đình", "Thần Thoại",
    "Bí Ẩn", "Tâm Lý", "Học Đường",
    "Hành Động", "Thể Thao", "Kinh Điển",
    "Hài Hước", "Võ Thuật", "Phim 18+",
    "Phiêu Lưu", "Cổ Trang", 
    "Kinh Dị", "Hình Sự"
]

# Định nghĩa loại phim
FILM_TYPES = ["Phim Lẻ", "Phim Bộ", "Phim Hoạt Hình"]

# Dictionary ánh xạ tiếng Việt không dấu sang có dấu
VIETNAMESE_MAPPING = {
    # Thể loại phim
    "hanh dong": "Hành Động",
    "tinh cam": "Tình Cảm",
    "hai huoc": "Hài Hước",
    "kinh di": "Kinh Dị", 
    "vien tuong": "Viễn Tưởng",
    "phieu luu": "Phiêu Lưu",
    "hoat hinh": "Hoạt Hình",
    "tai lieu": "Tài Liệu",
    "than thoai": "Thần Thoại",
    "co trang": "Cổ Trang",
    "hinh su": "Hình Sự",
    "tam ly": "Tâm Lý",
    "hoc duong": "Học Đường",
    "vo thuat": "Võ Thuật",
    "chien tranh": "Chiến Tranh",
    "am nhac": "Âm Nhạc",
    
    # Loại phim
    "phim le": "Phim Lẻ",
    "phim bo": "Phim Bộ",
    "phim hoat hinh": "Phim Hoạt Hình",
    
    # Ý định người dùng
    "de xuat": "đề xuất",
    "gioi thieu": "giới thiệu",
    "tim kiem": "tìm kiếm",
    "tim phim": "tìm phim",
    "xem phim": "xem phim",
    "thich": "thích",
    "khong thich": "không thích",
    "giup do": "giúp đỡ",
}

class InitDataToCSV:
    """Chuyển đổi dữ liệu từ MySQL sang CSV."""
    def __init__(self, db_config=None):
        if db_config is None:
            self.db_config = {
                "host": os.getenv('DB_HOST', 'localhost'),
                "user": os.getenv('DB_USER', 'root'),
                "password": os.getenv('DB_PASSWORD', ''),
                "database": os.getenv('DB_NAME', 'csdl_phim')
            }
        else:
            self.db_config = db_config

    def get_db_connection(self):
        """Thiết lập kết nối đến cơ sở dữ liệu MySQL."""
        try:
            return mysql.connector.connect(**self.db_config)
        except mysql.connector.Error as err:
            logger.error(f"Lỗi kết nối đến cơ sở dữ liệu: {err}")
            raise

    def load_movies_data(self):
        """Tải dữ liệu phim từ cơ sở dữ liệu MySQL."""
        try:
            conn = self.get_db_connection()
            query = """
            SELECT 
                p.id, 
                p.ten_phim AS title, 
                p.slug_phim AS slug, 
                p.hinh_anh AS hinh_anh, 
                p.dao_dien AS director, 
                p.quoc_gia AS country, 
                p.nam_san_xuat AS year, 
                lp.ten_loai_phim AS film_type,
                MAX(p.mo_ta) AS plot, 
                GROUP_CONCAT(DISTINCT tl.ten_the_loai SEPARATOR '|') AS genres
            FROM 
                phims p
            LEFT JOIN 
                chi_tiet_the_loais ctl ON p.id = ctl.id_phim
            LEFT JOIN 
                the_loais tl ON ctl.id_the_loai = tl.id
            LEFT JOIN 
                loai_phims lp ON p.id_loai_phim = lp.id
            WHERE 
                p.tinh_trang = 1
            GROUP BY 
                p.id, p.ten_phim
            """
            movies = pd.read_sql(query, conn)
            conn.close()
            return movies
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu phim: {e}")
            raise

    def preprocess_data(self, movies_df):
        """Tiền xử lý dữ liệu phim cho hệ thống đề xuất."""
        try:
            # Điền các giá trị thiếu
            movies_df['plot'] = movies_df['plot'].fillna('')
            movies_df['genres'] = movies_df['genres'].fillna('')
            movies_df['director'] = movies_df['director'].fillna('')
            movies_df['country'] = movies_df['country'].fillna('')
            movies_df['year'] = movies_df['year'].fillna(0)
            movies_df['film_type'] = movies_df['film_type'].fillna('')
            movies_df['hinh_anh'] = movies_df['hinh_anh'].fillna('')
            movies_df['slug'] = movies_df['slug'].fillna('')
            
            # Chuẩn hóa đường dẫn poster
            base_url = os.getenv('IMAGE_BASE_URL', 'http://localhost:5173/images/movies/')
            movies_df['poster_url'] = movies_df['hinh_anh'].apply(
                lambda x: f"{base_url}{x}" if x else f"https://via.placeholder.com/500x300?text=No+Image"
            )
            
            # Tạo đặc trưng kết hợp cho tính toán độ tương đồng
            movies_df['combined_features'] = (
                movies_df['title'] + ' ' + 
                movies_df['plot'] + ' ' + 
                movies_df['genres'] + ' ' + 
                movies_df['director'] + ' ' + 
                movies_df['country'] + ' ' + 
                movies_df['film_type']
            )
            
            return movies_df
        except Exception as e:
            logger.error(f"Lỗi khi tiền xử lý dữ liệu: {e}")
            raise

    def save_to_csv(self, movies_df, output_path='vietnamese_movies.csv'):
        """Lưu dữ liệu phim đã xử lý vào file CSV."""
        try:
            # Lưu vào CSV đầy đủ (bao gồm cả combined_features)
            movies_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Đã lưu dữ liệu thành công vào {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Lỗi khi lưu vào CSV: {e}")
            raise

    def process_and_save(self, output_path='vietnamese_movies.csv', force_update=False):
        """Phương thức chính để xử lý dữ liệu và lưu vào CSV."""
        try:
            # Kiểm tra file đã tồn tại và không force update
            if os.path.exists(output_path) and not force_update:
                logger.info(f"File CSV đã tồn tại và không yêu cầu cập nhật lại: {output_path}")
                return output_path
                
            # Tải dữ liệu từ cơ sở dữ liệu
            movies_df = self.load_movies_data()
            logger.info(f"Đã tải {len(movies_df)} bộ phim từ cơ sở dữ liệu")
            
            # Tiền xử lý dữ liệu
            processed_df = self.preprocess_data(movies_df)
            logger.info("Đã hoàn thành tiền xử lý dữ liệu")
            
            # Lưu vào CSV
            csv_path = self.save_to_csv(processed_df, output_path)
            logger.info(f"Quá trình chuyển đổi dữ liệu thành công. File CSV: {csv_path}")
            return csv_path
        except Exception as e:
            logger.error(f"Lỗi trong quá trình xử lý và lưu dữ liệu: {e}")
            raise

    @staticmethod
    def create_sample_csv(output_path='vietnamese_movies.csv'):
        """Tạo file CSV mẫu nếu không thể kết nối đến database."""
        try:
            # Tạo dữ liệu phim mẫu
            movies_data = [
                {
                    "id": 1,
                    "title": "Phim Hành Động Mẫu",
                    "slug": "phim-hanh-dong-mau",
                    "hinh_anh": "hanh-dong-mau.jpg",
                    "poster_url": "https://via.placeholder.com/500x300?text=Phim+H%C3%A0nh+%C4%90%E1%BB%99ng+M%E1%BA%ABu",
                    "director": "Đạo diễn A",
                    "country": "Việt Nam",
                    "year": 2023,
                    "film_type": "Phim Lẻ",
                    "plot": "Một bộ phim hành động gay cấn với những pha hành động mãn nhãn và tình tiết hấp dẫn.",
                    "genres": "Hành Động|Phiêu Lưu",
                    "combined_features": "Phim Hành Động Mẫu Một bộ phim hành động gay cấn với những pha hành động mãn nhãn và tình tiết hấp dẫn. Hành Động|Phiêu Lưu Đạo diễn A Việt Nam 2023 Phim Lẻ"
                },
                {
                    "id": 2,
                    "title": "Phim Tình Cảm Mẫu",
                    "slug": "phim-tinh-cam-mau",
                    "hinh_anh": "tinh-cam-mau.jpg",
                    "poster_url": "https://via.placeholder.com/500x300?text=Phim+T%C3%ACnh+C%E1%BA%A3m+M%E1%BA%ABu",
                    "director": "Đạo diễn B",
                    "country": "Hàn Quốc",
                    "year": 2022,
                    "film_type": "Phim Bộ",
                    "plot": "Câu chuyện tình yêu đầy cảm động giữa hai nhân vật chính với nhiều thử thách và khó khăn.",
                    "genres": "Tình Cảm|Tâm Lý",
                    "combined_features": "Phim Tình Cảm Mẫu Câu chuyện tình yêu đầy cảm động giữa hai nhân vật chính với nhiều thử thách và khó khăn. Tình Cảm|Tâm Lý Đạo diễn B Hàn Quốc 2022 Phim Bộ"
                },
                {
                    "id": 3,
                    "title": "Hoạt Hình Mẫu",
                    "slug": "hoat-hinh-mau",
                    "hinh_anh": "hoat-hinh-mau.jpg",
                    "poster_url": "https://via.placeholder.com/500x300?text=Ho%E1%BA%A1t+H%C3%ACnh+M%E1%BA%ABu",
                    "director": "Đạo diễn C",
                    "country": "Nhật Bản",
                    "year": 2021,
                    "film_type": "Phim Hoạt Hình",
                    "plot": "Một cuộc phiêu lưu kỳ thú trong thế giới hoạt hình với những nhân vật dễ thương và bài học ý nghĩa.",
                    "genres": "Hoạt Hình|Gia Đình",
                    "combined_features": "Hoạt Hình Mẫu Một cuộc phiêu lưu kỳ thú trong thế giới hoạt hình với những nhân vật dễ thương và bài học ý nghĩa. Hoạt Hình|Gia Đình Đạo diễn C Nhật Bản 2021 Phim Hoạt Hình"
                },
                {
                    "id": 4,
                    "title": "Phim Kinh Dị Mẫu",
                    "slug": "phim-kinh-di-mau",
                    "hinh_anh": "kinh-di-mau.jpg",
                    "poster_url": "https://via.placeholder.com/500x300?text=Phim+Kinh+D%E1%BB%8B+M%E1%BA%ABu",
                    "director": "Đạo diễn D",
                    "country": "Mỹ",
                    "year": 2020,
                    "film_type": "Phim Lẻ",
                    "plot": "Những sự kiện kinh hoàng diễn ra trong một ngôi nhà bỏ hoang khiến nhóm bạn trẻ phải đối mặt với nỗi sợ hãi tột cùng.",
                    "genres": "Kinh Dị|Hồi hộp",
                    "combined_features": "Phim Kinh Dị Mẫu Những sự kiện kinh hoàng diễn ra trong một ngôi nhà bỏ hoang khiến nhóm bạn trẻ phải đối mặt với nỗi sợ hãi tột cùng. Kinh Dị|Hồi hộp Đạo diễn D Mỹ 2020 Phim Lẻ"
                },
                {
                    "id": 5,
                    "title": "Phim Cổ Trang Mẫu",
                    "slug": "phim-co-trang-mau",
                    "hinh_anh": "co-trang-mau.jpg",
                    "poster_url": "https://via.placeholder.com/500x300?text=Phim+C%E1%BB%95+Trang+M%E1%BA%ABu",
                    "director": "Đạo diễn E",
                    "country": "Trung Quốc",
                    "year": 2019,
                    "film_type": "Phim Bộ",
                    "plot": "Câu chuyện về cuộc đời của một vị tướng tài ba trong thời loạn lạc, với những chiến công hiển hách và tình yêu sâu đậm.",
                    "genres": "Cổ Trang|Võ Thuật|Lịch Sử",
                    "combined_features": "Phim Cổ Trang Mẫu Câu chuyện về cuộc đời của một vị tướng tài ba trong thời loạn lạc, với những chiến công hiển hách và tình yêu sâu đậm. Cổ Trang|Võ Thuật|Lịch Sử Đạo diễn E Trung Quốc 2019 Phim Bộ"
                }
            ]
            
            # Tạo DataFrame và lưu vào CSV
            df = pd.DataFrame(movies_data)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Đã tạo file CSV mẫu với {len(df)} bộ phim tại: {output_path}")
            return df
        except Exception as e:
            logger.error(f"Lỗi khi tạo file CSV mẫu: {str(e)}")
            raise


class MovieRecommender:
    """Hệ thống đề xuất phim từ dữ liệu CSV."""
    def __init__(self, csv_path):
        """Khởi tạo hệ thống đề xuất phim từ file CSV."""
        self.df = self.load_and_preprocess_data(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.features = None
        self.similarity_matrix = None
        self.create_features()
        self.map_indices()
    
    def load_and_preprocess_data(self, csv_path):
        """Đọc file CSV và tiền xử lý dữ liệu."""
        try:
            # Kiểm tra file CSV
            if not os.path.exists(csv_path):
                logger.warning(f"File CSV {csv_path} không tồn tại. Tạo file mẫu.")
                InitDataToCSV.create_sample_csv(csv_path)
                
            # Đọc file CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Đã đọc file CSV thành công, có {len(df)} dòng dữ liệu")
            
            # Điền các giá trị NaN
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].fillna('')
                else:
                    df[col] = df[col].fillna(0)
            
            # Kiểm tra nếu chưa có cột combined_features
            if 'combined_features' not in df.columns:
                logger.info("Tạo cột combined_features cho phim")
                df['combined_features'] = self.create_combined_features(df)
            
            # Kiểm tra nếu chưa có cột poster_url
            if 'poster_url' not in df.columns:
                logger.info("Tạo cột poster_url cho phim")
                # base_url = os.getenv('IMAGE_BASE_URL', 'http://localhost:5173/images/movies/')
                df['poster_url'] = df['hinh_anh']
            
            return df
        except Exception as e:
            logger.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
            raise
    
    def create_combined_features(self, df):
        """Tạo chuỗi đặc trưng kết hợp cho mỗi bộ phim."""
        features = []
        
        for _, row in df.iterrows():
            combined = ""
            
            # Tạo feature với trọng số khác nhau cho các thành phần
            if 'title' in df.columns and row['title']:
                combined += f"{row['title']} {row['title']} {row['title']} "  # Trọng số cao nhất cho tiêu đề
            
            if 'genres' in df.columns and row['genres']:
                genres = str(row['genres']).replace('|', ' ')
                combined += f"{genres} {genres} "  # Trọng số cao cho thể loại
            
            if 'plot' in df.columns and row['plot']:
                combined += f"{row['plot']} "
                
            if 'director' in df.columns and row['director']:
                combined += f"{row['director']} "
                
            if 'actors' in df.columns and row['actors']:
                combined += f"{row['actors']} "
                
            if 'film_type' in df.columns and row['film_type']:
                combined += f"{row['film_type']} "
                
            if 'country' in df.columns and row['country']:
                combined += f"{row['country']} "
                
            if 'year' in df.columns and row['year']:
                combined += f"{row['year']} "
            
            features.append(combined.strip())
        
        return features
    
    def create_features(self):
        """Tạo đặc trưng TF-IDF và ma trận tương đồng."""
        try:
            # Kiểm tra nếu 'combined_features' tồn tại trong DataFrame
            if 'combined_features' not in self.df.columns:
                logger.error("Cột 'combined_features' không tồn tại trong dữ liệu!")
                # Tạo cột combined_features nếu chưa có
                self.df['combined_features'] = self.create_combined_features(self.df)
                
            # Tạo đặc trưng TF-IDF
            logger.info("Đang tạo đặc trưng TF-IDF...")
            
            # Đảm bảo combined_features là chuỗi
            self.df['combined_features'] = self.df['combined_features'].astype(str)
            
            # Tính toán ma trận đặc trưng
            self.features = self.vectorizer.fit_transform(self.df['combined_features'])
            
            # Tính toán ma trận tương đồng
            logger.info("Đang tính toán ma trận tương đồng...")
            self.similarity_matrix = cosine_similarity(self.features)
            
            logger.info(f"Kích thước ma trận đặc trưng: {self.features.shape}")
            logger.info(f"Kích thước ma trận tương đồng: {self.similarity_matrix.shape}")
        except Exception as e:
            logger.error(f"Lỗi khi tạo đặc trưng: {str(e)}")
            raise
    
    def map_indices(self):
        """Tạo ánh xạ giữa tiêu đề phim và chỉ số."""
        try:
            # Tạo bảng tra cứu chỉ số của phim theo tiêu đề
            self.title_to_index = {}
            for i, title in enumerate(self.df['title']):
                # Bảng tra cứu chính xác
                self.title_to_index[title.lower()] = i
                
                # Từng phần của tên phim (để tìm kiếm mờ)
                words = title.lower().split()
                for j in range(len(words)):
                    # Tạo từ khóa tìm kiếm từ các từ liên tiếp
                    keyword = " ".join(words[j:min(j+3, len(words))])
                    if len(keyword) > 3 and keyword not in self.title_to_index:  # từ khóa có ít nhất 4 ký tự
                        self.title_to_index[keyword] = i
        except Exception as e:
            logger.error(f"Lỗi khi tạo ánh xạ chỉ số: {str(e)}")
            self.title_to_index = {}
    
    @lru_cache(maxsize=100)
    def get_movie_index(self, movie_title):
        """Lấy chỉ số của phim theo tiêu đề."""
        try:
            # Tìm kiếm trong bảng tra cứu
            movie_title_lower = movie_title.lower()
            
            # Tìm kiếm chính xác
            if movie_title_lower in self.title_to_index:
                return self.title_to_index[movie_title_lower]
            
            # Tìm kiếm một phần
            for key, index in self.title_to_index.items():
                if movie_title_lower in key or key in movie_title_lower:
                    return index
            
            # Tìm kiếm theo cách truyền thống nếu bảng tra cứu thất bại
            matches = self.df[self.df['title'].str.lower() == movie_title_lower]
            if not matches.empty:
                return matches.index[0]
            
            # Tìm kiếm một phần
            matches = self.df[self.df['title'].str.lower().str.contains(movie_title_lower, na=False)]
            if not matches.empty:
                return matches.index[0]
            
            return None
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm phim: {str(e)}")
            return None
    
    def get_recommendations(self, movie_title=None, genres=None, film_type=None, top_n=5):
        """Lấy đề xuất phim dựa trên tiêu đề, thể loại hoặc loại phim."""
        if movie_title:
            return self.get_similar_movies(movie_title, top_n)
        elif genres:
            return self.get_genre_recommendations(genres, film_type, top_n)
        elif film_type:
            return self.get_film_type_recommendations(film_type, top_n)
        else:
            return self.get_popular_movies(top_n)
    
    def get_similar_movies(self, movie_title, top_n=5):
        start_time = time.time()
        logger.info(f"Tìm phim tương tự với '{movie_title}'")
        
        try:
            # Thử tìm chỉ số của phim
            idx = self.get_movie_index(movie_title)
            
            # Nếu không tìm thấy, thử tìm kiếm mờ
            if idx is None:
                logger.info(f"Không tìm thấy phim chính xác: {movie_title}, thử tìm kiếm mờ...")
                search_results = self.search_movies(movie_title, 1)
                
                if search_results:
                    found_movie = search_results[0]
                    movie_title = found_movie['title']
                    idx = self.get_movie_index(movie_title)
                    logger.info(f"Đã tìm thấy phim gần đúng: {movie_title}")
                else:
                    logger.warning(f"Không tìm thấy phim nào với tên: {movie_title}")
                    return []
            
            if idx is None:
                logger.warning(f"Vẫn không thể tìm thấy chỉ số cho phim: {movie_title}")
                return []
            
            # Lấy điểm tương đồng
            similarity_scores = list(enumerate(self.similarity_matrix[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Debug info
            logger.info(f"Top 3 phim tương tự với {movie_title}:")
            for i, (movie_idx, score) in enumerate(similarity_scores[1:4]):
                logger.info(f"{i+1}. {self.df.iloc[movie_idx]['title']} - Score: {score:.4f}")
            
            # Lấy top N phim tương tự (loại trừ phim đầu vào)
            similar_movies_indices = [i[0] for i in similarity_scores[1:top_n+1]]
            
            # Trả về thông tin chi tiết của phim
            recommendations = []
            for idx in similar_movies_indices:
                movie_data = self.df.iloc[idx].to_dict()
                # Loại bỏ combined_features khỏi đầu ra
                if 'combined_features' in movie_data:
                    del movie_data['combined_features']
                # Thêm điểm tương đồng
                movie_data['similarity_score'] = float(similarity_scores[similar_movies_indices.index(idx) + 1][1])
                recommendations.append(movie_data)
            
            logger.info(f"Thời gian tìm phim tương tự: {time.time() - start_time:.4f}s")
            return recommendations
        except Exception as e:
            logger.error(f"Lỗi khi lấy phim tương tự: {str(e)}")
            return []
    
    def _calculate_string_similarity(self, str1, str2):
        """Tính độ tương đồng giữa hai chuỗi."""
        # Phương pháp đơn giản: tỷ lệ ký tự chung
        if not str1 or not str2:
            return 0
        
        # Chuyển đổi cả hai chuỗi thành tập hợp ký tự
        set1 = set(str1)
        set2 = set(str2)
        
        # Tính giao và hợp của hai tập hợp
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        # Trả về tỷ lệ phần trăm ký tự trùng nhau
        return intersection / union if union > 0 else 0

    def get_genre_recommendations(self, genres, film_type=None, top_n=5):
        """Lấy đề xuất dựa trên thể loại và loại phim (nếu có)."""
        start_time = time.time()
        try:
            # Đảm bảo genres là list
            if isinstance(genres, str):
                genres = [genres]
            
            # Lọc các thể loại rỗng
            genres = [g.strip() for g in genres if g and g.strip()]
            
            if not genres:
                logger.warning("Không có thể loại hợp lệ được cung cấp")
                return self.get_popular_movies(top_n)
            
            # Chuẩn hóa thể loại - xử lý tiếng Việt không dấu và lỗi chính tả
            normalized_genres = []
            for genre in genres:
                # Kiểm tra nếu đã là thể loại hợp lệ
                if genre in VIETNAMESE_GENRES:
                    normalized_genres.append(genre)
                    continue
                    
                # Kiểm tra ánh xạ tiếng Việt không dấu
                lower_genre = genre.lower()
                if lower_genre in VIETNAMESE_MAPPING:
                    normalized_genre = VIETNAMESE_MAPPING[lower_genre]
                    logger.info(f"Đã chuẩn hóa thể loại '{genre}' thành '{normalized_genre}'")
                    normalized_genres.append(normalized_genre)
                    continue
                
                # Thử tìm thể loại gần đúng nhất
                best_match = None
                max_similarity = 0
                
                for vn_genre in VIETNAMESE_GENRES:
                    # So sánh không phân biệt hoa thường
                    similarity = 0
                    try:
                        # Nếu có phương thức tính tương đồng
                        if hasattr(self, '_calculate_string_similarity'):
                            similarity = self._calculate_string_similarity(lower_genre, vn_genre.lower())
                    except Exception as e:
                        logger.error(f"Lỗi khi tính tương đồng: {e}")
                        continue
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = vn_genre
                
                if best_match and max_similarity > 0.5:  # Ngưỡng tương đồng
                    logger.info(f"Đã tìm thấy thể loại gần đúng: '{best_match}' cho '{genre}'")
                    normalized_genres.append(best_match)
                else:
                    # Nếu không tìm thấy thể loại gần đúng, thêm thể loại gốc
                    logger.warning(f"Không thể chuẩn hóa thể loại: '{genre}', sử dụng nguyên gốc")
                    normalized_genres.append(genre)
            
            logger.info(f"Tìm phim theo thể loại đã chuẩn hóa: {normalized_genres}, loại phim: {film_type}")
            
            # Kiểm tra và log cấu trúc dữ liệu
            if 'genres' in self.df.columns:
                # Lấy mẫu các giá trị thể loại để debug
                sample_genres = self.df['genres'].dropna().sample(min(5, len(self.df))).tolist()
                logger.info(f"Mẫu thể loại trong cơ sở dữ liệu: {sample_genres}")
            else:
                logger.warning("Không tìm thấy cột 'genres' trong DataFrame")
                return self.get_popular_movies(top_n)
            
            # Tính điểm trùng khớp thể loại cho mỗi bộ phim với so sánh linh hoạt hơn
            genre_scores = []
            for idx, row in self.df.iterrows():
                if 'genres' not in self.df.columns or pd.isna(row['genres']) or not row['genres']:
                    continue
                
                # Kiểm tra loại phim nếu được chỉ định
                if film_type and 'film_type' in self.df.columns and pd.notna(row['film_type']):
                    if row['film_type'] != film_type:
                        continue
                
                # Chuyển đổi chuỗi genre thành danh sách
                movie_genres_raw = []
                if isinstance(row['genres'], str):
                    # Thử các dấu phân cách khác nhau
                    if '|' in row['genres']:
                        movie_genres_raw = row['genres'].split('|')
                    elif ',' in row['genres']:
                        movie_genres_raw = row['genres'].split(',')
                    elif ';' in row['genres']:
                        movie_genres_raw = row['genres'].split(';')
                    else:
                        # Nếu không có dấu phân cách, xem như một thể loại
                        movie_genres_raw = [row['genres']]
                else:
                    continue
                
                # Làm sạch các thể loại phim
                movie_genres = [g.strip() for g in movie_genres_raw if g and g.strip()]
                
                # Tính điểm trùng khớp với nhiều phương pháp so sánh
                match_score = 0
                for user_genre in normalized_genres:
                    user_genre_lower = user_genre.lower()
                    
                    for movie_genre in movie_genres:
                        movie_genre_lower = movie_genre.lower()
                        
                        # So sánh chính xác
                        if user_genre == movie_genre:
                            match_score += 1
                            break
                        
                        # So sánh không phân biệt hoa thường
                        elif user_genre_lower == movie_genre_lower:
                            match_score += 0.9
                            break
                        
                        # So sánh chứa từ khóa
                        elif user_genre_lower in movie_genre_lower or movie_genre_lower in user_genre_lower:
                            match_score += 0.8
                            break
                        
                        # So sánh từng từ
                        elif any(word in movie_genre_lower.split() for word in user_genre_lower.split()):
                            match_score += 0.7
                            break
                
                if match_score > 0:
                    # Thêm trọng số nếu phim có rating cao
                    rating_weight = 1.0
                    if 'rating' in row and pd.notna(row['rating']):
                        try:
                            rating = float(row['rating'])
                            rating_weight = 1.0 + (rating / 10.0)
                        except (ValueError, TypeError):
                            pass
                    
                    # Thêm trọng số nếu phim mới
                    year_weight = 1.0
                    current_year = datetime.datetime.now().year
                    if 'year' in row and pd.notna(row['year']):
                        try:
                            year = int(row['year'])
                            if year > 0:
                                age = current_year - year
                                year_weight = 1.0 + max(0, (10 - age) / 10.0)
                        except (ValueError, TypeError):
                            pass
                    
                    # Điểm cuối cùng
                    final_score = match_score * rating_weight * year_weight
                    genre_scores.append((idx, final_score))
            
            # Sắp xếp theo điểm trùng khớp
            genre_scores = sorted(genre_scores, key=lambda x: x[1], reverse=True)
            
            logger.info(f"Số lượng phim phù hợp: {len(genre_scores)}")
            
            # Nếu không có kết quả, thử tìm với cách khác
            if not genre_scores:
                logger.warning(f"Không tìm thấy phim phù hợp với thể loại {normalized_genres} và loại phim {film_type}")
                
                # Thử tìm kiếm theo từng thể loại riêng lẻ
                logger.info("Thử tìm kiếm theo từng thể loại riêng lẻ...")
                all_matches = []
                
                for genre in normalized_genres:
                    # Tìm kiếm tất cả phim có chứa thể loại này
                    for idx, row in self.df.iterrows():
                        if 'genres' not in self.df.columns or pd.isna(row['genres']) or not row['genres']:
                            continue
                        
                        # Kiểm tra loại phim nếu được chỉ định
                        if film_type and 'film_type' in self.df.columns and pd.notna(row['film_type']):
                            if row['film_type'] != film_type:
                                continue
                        
                        # Tìm kiếm không phân biệt hoa thường
                        if isinstance(row['genres'], str) and genre.lower() in row['genres'].lower():
                            all_matches.append((idx, 1.0))  # Điểm mặc định
                
                if all_matches:
                    logger.info(f"Tìm thấy {len(all_matches)} phim sau khi tìm kiếm linh hoạt")
                    # Loại bỏ trùng lặp và lấy top N
                    unique_matches = list(set(all_matches))
                    top_matches = unique_matches[:top_n]
                    
                    # Trả về thông tin chi tiết của phim
                    recommendations = []
                    for idx, score in top_matches:
                        movie_data = self.df.iloc[idx].to_dict()
                        
                        # Thêm điểm tương đồng vào kết quả
                        movie_data['similarity_score'] = float(score)
                        
                        # Loại bỏ combined_features khỏi đầu ra
                        if 'combined_features' in movie_data:
                            del movie_data['combined_features']
                        
                        recommendations.append(movie_data)
                    
                    return recommendations
                
                # Nếu vẫn không tìm thấy, trả về phim phổ biến
                return self.get_popular_movies(top_n)
            
            # Lấy top N kết quả trùng khớp
            top_matches = genre_scores[:top_n]
            
            # Trả về thông tin chi tiết của phim
            recommendations = []
            for idx, score in top_matches:
                movie_data = self.df.iloc[idx].to_dict()
                
                # Thêm điểm tương đồng vào kết quả
                movie_data['similarity_score'] = float(score)
                
                # Loại bỏ combined_features khỏi đầu ra
                if 'combined_features' in movie_data:
                    del movie_data['combined_features']
                
                recommendations.append(movie_data)
            
            logger.info(f"Thời gian tìm phim theo thể loại: {time.time() - start_time:.4f}s")
            return recommendations
        except Exception as e:
            logger.error(f"Lỗi khi lấy đề xuất theo thể loại: {str(e)}")
            return self.get_popular_movies(top_n)

    def get_film_type_recommendations(self, film_type, top_n=5):
        """Lấy đề xuất dựa trên loại phim."""
        try:
            if 'film_type' not in self.df.columns:
                logger.warning("Không có cột 'film_type' trong dữ liệu")
                return self.get_popular_movies(top_n)
            
            # Lọc theo loại phim
            filtered_df = self.df[self.df['film_type'] == film_type]
            
            # Nếu không có phim thuộc loại này, trả về phim phổ biến
            if filtered_df.empty:
                logger.warning(f"Không tìm thấy phim loại '{film_type}'")
                return self.get_popular_movies(top_n)
            
            # Sắp xếp theo đánh giá và năm phát hành
            if 'rating' in filtered_df.columns and 'year' in filtered_df.columns:
                # Tạo cột điểm tổng hợp từ rating và year
                current_year = datetime.datetime.now().year
                
                # Chuyển đổi rating và year sang số
                filtered_df['rating_num'] = pd.to_numeric(filtered_df['rating'], errors='coerce').fillna(0)
                filtered_df['year_num'] = pd.to_numeric(filtered_df['year'], errors='coerce').fillna(0)
                
                # Tính trọng số năm (phim càng mới điểm càng cao)
                filtered_df['year_weight'] = filtered_df['year_num'].apply(
                    lambda y: 1.0 + max(0, (10 - (current_year - y if y > 0 else 100)) / 10.0)
                )
                
                # Tính điểm tổng hợp
                filtered_df['combined_score'] = filtered_df['rating_num'] * filtered_df['year_weight']
                
                # Sắp xếp theo điểm tổng hợp
                filtered_df = filtered_df.sort_values(by='combined_score', ascending=False)
            elif 'rating' in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by='rating', ascending=False)
            elif 'year' in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by='year', ascending=False)
            
            # Lấy top N phim
            top_movies = filtered_df.head(top_n)
            
            # Trả về thông tin chi tiết
            recommendations = []
            for _, row in top_movies.iterrows():
                movie_data = row.to_dict()
                # Loại bỏ combined_features và các cột tạm thời khỏi đầu ra
                for col in ['combined_features', 'rating_num', 'year_num', 'year_weight', 'combined_score']:
                    if col in movie_data:
                        del movie_data[col]
                recommendations.append(movie_data)
            
            return recommendations
        except Exception as e:
            logger.error(f"Lỗi khi lấy đề xuất theo loại phim: {str(e)}")
            return self.get_popular_movies(top_n)

    def get_popular_movies(self, top_n=5):
        """Lấy các phim phổ biến dựa trên đánh giá hoặc ngẫu nhiên."""
        try:
            # Kiểm tra số lượng phim
            if len(self.df) < top_n:
                return [row.to_dict() for _, row in self.df.iterrows()]
            
            # Quyết định dựa trên rating hoặc ngẫu nhiên
            use_random = random.random() < 0.3  # 30% cơ hội sử dụng ngẫu nhiên
            
            if use_random:
                # Chọn ngẫu nhiên
                logger.info("Chọn phim ngẫu nhiên")
                random_indices = np.random.choice(len(self.df), top_n, replace=False)
                top_movies = self.df.iloc[random_indices]
            else:
                # Sắp xếp theo rating nếu có
                if 'rating' in self.df.columns:
                    logger.info("Chọn phim theo rating")
                    filtered_df = self.df[self.df['rating'].notna()]
                    if not filtered_df.empty:
                        # Thêm một chút ngẫu nhiên vào sắp xếp để không luôn trả về cùng một kết quả
                        filtered_df['random'] = np.random.uniform(0.9, 1.1, len(filtered_df))
                        filtered_df['adjusted_rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce') * filtered_df['random']
                        top_movies = filtered_df.sort_values(by='adjusted_rating', ascending=False).head(top_n)
                        # Xóa cột tạm
                        top_movies = top_movies.drop(columns=['random', 'adjusted_rating'])
                    else:
                        # Nếu không có rating, chọn ngẫu nhiên
                        random_indices = np.random.choice(len(self.df), top_n, replace=False)
                        top_movies = self.df.iloc[random_indices]
                else:
                    # Nếu không có cột rating, ưu tiên phim mới
                    if 'year' in self.df.columns:
                        logger.info("Chọn phim theo năm phát hành")
                        filtered_df = self.df[self.df['year'].notna()]
                        if not filtered_df.empty:
                            # Thêm một chút ngẫu nhiên
                            filtered_df['random'] = np.random.uniform(0.9, 1.1, len(filtered_df))
                            filtered_df['adjusted_year'] = pd.to_numeric(filtered_df['year'], errors='coerce') * filtered_df['random']
                            top_movies = filtered_df.sort_values(by='adjusted_year', ascending=False).head(top_n)
                            # Xóa cột tạm
                            top_movies = top_movies.drop(columns=['random', 'adjusted_year'])
                        else:
                            # Nếu không có năm, chọn ngẫu nhiên
                            random_indices = np.random.choice(len(self.df), top_n, replace=False)
                            top_movies = self.df.iloc[random_indices]
                    else:
                        # Nếu không có cả rating và year, chọn ngẫu nhiên
                        random_indices = np.random.choice(len(self.df), top_n, replace=False)
                        top_movies = self.df.iloc[random_indices]
            
            # Trả về thông tin chi tiết
            recommendations = []
            for _, row in top_movies.iterrows():
                movie_data = row.to_dict()
                # Loại bỏ combined_features khỏi đầu ra
                if 'combined_features' in movie_data:
                    del movie_data['combined_features']
                recommendations.append(movie_data)
            
            return recommendations
        except Exception as e:
            logger.error(f"Lỗi khi lấy phim phổ biến: {str(e)}")
            
            # Fallback: trả về ngẫu nhiên nếu có lỗi
            try:
                random_indices = np.random.choice(len(self.df), min(top_n, len(self.df)), replace=False)
                random_movies = self.df.iloc[random_indices]
                
                recommendations = []
                for _, row in random_movies.iterrows():
                    movie_data = row.to_dict()
                    if 'combined_features' in movie_data:
                        del movie_data['combined_features']
                    recommendations.append(movie_data)
                
                return recommendations
            except:
                # Nếu vẫn lỗi, trả về list rỗng
                return []

    def search_movies(self, query, top_n=5):
        """Tìm kiếm phim theo tiêu đề, diễn viên, đạo diễn hoặc thể loại."""
        try:
            query = query.lower()
            
            # Tìm kiếm trong các cột khác nhau
            results = []
            
            # Các cột tìm kiếm và trọng số tương ứng
            search_columns = {
                'title': 3.0,       # Tiêu đề có trọng số cao nhất
                'genres': 2.0,       # Thể loại cũng quan trọng
                'director': 1.5,    # Đạo diễn
                'actors': 1.2,      # Diễn viên
                'plot': 1.0,        # Nội dung
                'country': 0.7,     # Quốc gia
                'film_type': 0.8    # Loại phim
            }
            
            # Tìm kiếm trong từng cột và tính điểm
            for idx, row in self.df.iterrows():
                score = 0.0
                matches = False
                
                for col, weight in search_columns.items():
                    if col in self.df.columns and pd.notna(row[col]) and query in str(row[col]).lower():
                        score += weight
                        matches = True
                
                # Nếu có kết quả trùng khớp, thêm vào danh sách kết quả
                if matches:
                    movie_data = row.to_dict()
                    movie_data['search_score'] = score
                    
                    # Loại bỏ combined_features khỏi đầu ra
                    if 'combined_features' in movie_data:
                        del movie_data['combined_features']
                    
                    results.append(movie_data)
            
            # Sắp xếp kết quả theo điểm tìm kiếm
            results = sorted(results, key=lambda x: x['search_score'], reverse=True)
            
            # Lấy top N kết quả
            return results[:top_n]
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm phim: {str(e)}")
            return []


class AIMovieChatbot:
    """Chatbot đề xuất phim thông minh sử dụng Gemini AI."""
    def __init__(self, csv_path, api_key):
        """Khởi tạo chatbot với hệ thống đề xuất phim và Gemini AI."""
        self.recommender = MovieRecommender(csv_path)
        self.google_api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.history = []
        self.user_preferences = {
            "genres": [],
            "film_types": [],
            "liked_movies": [],
            "disliked_movies": [],
            "mentioned_movies": []
        }
        self.conversation_context = "movie"  # movie or vip
        
        # Tạo prompts mẫu
        self._create_system_prompts()
    
    def _create_system_prompts(self):
        """Tạo các system prompts cho các ngữ cảnh khác nhau."""
        # Prompt gốc cho phim
        self.movie_system_prompt = """
        Bạn là trợ lý đề xuất phim thông minh, chuyên giúp người dùng tìm kiếm và khám phá phim.
        
        Hãy trả lời chi tiết, tự nhiên và hữu ích, tránh các câu trả lời ngắn gọn một câu.
        Khi đề xuất phim, hãy giải thích lý do tại sao bạn đề xuất các phim đó.
        Khi nói về phim, hãy cung cấp thông tin phong phú về cốt truyện, diễn viên, đạo diễn, và đánh giá.
        
        Luôn trả lời bằng tiếng Việt, sử dụng ngôn ngữ thân thiện và tự nhiên.
        Không viết quá nhiều cảm thán từ hay biểu tượng cảm xúc.
        
        Nếu bạn không có thông tin về một phim cụ thể, hãy nói rằng bạn không có thông tin về phim đó nhưng có thể đề xuất các phim tương tự dựa trên mô tả.
        """
        
        # Prompt cho VIP
        self.vip_system_prompt = """
        Bạn là trợ lý nâng cấp tài khoản VIP cho website xem phim trực tuyến.  
        Trang web sử dụng hệ thống AI gợi ý phim thông minh WOPAI, mang đến trải nghiệm cá nhân hóa tốt nhất.

        Nhiệm vụ của bạn:
        - Giới thiệu về các quyền lợi của thành viên VIP.
        - Hướng dẫn người dùng đăng ký các gói VIP phù hợp.
        - Trả lời câu hỏi liên quan đến đăng ký/hủy gói.
        - Cung cấp thông tin liên hệ hỗ trợ: vietvo371@gmail.com

        Quyền lợi khi nâng cấp VIP:
        - Xem không giới hạn: Truy cập toàn bộ nội dung, bao gồm phim độc quyền.
        - Không quảng cáo: Trải nghiệm liền mạch, không bị gián đoạn.
        - Chất lượng cao: Xem phim với độ phân giải tốt nhất.
        - Hỗ trợ kỹ thuật 24/7: Ưu tiên hỗ trợ khi gặp sự cố.

        Bảng giá các gói VIP:
        - Gói 1 tháng: 25.000 VND (giảm còn 20.000 VND)
        * 1 tháng xem không giới hạn.
        * Không quảng cáo.
        * Chất lượng video cao nhất.
        * Hỗ trợ kỹ thuật ưu tiên.

        - Gói 3 tháng: 75.000 VND (giảm còn 60.000 VND)
        * 3 tháng xem không giới hạn.
        * Không quảng cáo.
        * Chất lượng video cao nhất.
        * Hỗ trợ kỹ thuật ưu tiên.

        - Gói 6 tháng: 150.000 VND (giảm còn 120.000 VND)
        * 6 tháng xem không giới hạn.
        * Không quảng cáo.
        * Chất lượng video cao nhất.
        * Hỗ trợ kỹ thuật ưu tiên.

        Câu hỏi thường gặp:
        - Làm sao để trở thành thành viên VIP?
        - Tôi có thể hủy đăng ký VIP bất kỳ lúc nào không?

        Quy tắc trả lời:
        - Trả lời rõ ràng, mạch lạc.
        - Định dạng trả lời dễ đọc (Markdown hoặc plain text).
        - Giao tiếp lịch sự, thân thiện, hỗ trợ tận tình.

        Ghi chú: Nếu cần thêm hỗ trợ, hãy liên hệ qua email: vietvo371@gmail.com
        """


    def process_message(self, message):
        """Xử lý tin nhắn người dùng với tăng cường Gemini."""
        try:
            # Lưu tin nhắn vào lịch sử
            self.history.append({"role": "user", "content": message})
            
            # Sử dụng Gemini để cải thiện việc xử lý đầu vào
            enhanced_analysis = self._enhance_input_with_gemini(message)
            logger.info(f"Gemini đã phân tích tin nhắn: {json.dumps(enhanced_analysis, ensure_ascii=False)}")
            
            # Cập nhật ngữ cảnh dựa trên phân tích nâng cao
            context_type = enhanced_analysis.get("context_type", "movie")
            if context_type in ["vip", "subscription", "payment"]:
                self.conversation_context = "vip"
            else:
                self.conversation_context = "movie"
            
            # Xử lý dựa trên ngữ cảnh
            if self.conversation_context == "vip":
                response_data = self._handle_vip_context(message, enhanced_analysis)
            else:
                response_data = self._handle_enhanced_movie_context(message, enhanced_analysis)
            
            # Xử lý đầu ra với Gemini để làm cho phản hồi tự nhiên hơn
            final_response = self._enhance_output_with_gemini(response_data)
            
            # Lưu phản hồi vào lịch sử
            self.history.append({"role": "assistant", "content": final_response})
            
            return final_response
        except Exception as e:
            error_msg = f"Đã xảy ra lỗi khi xử lý tin nhắn: {str(e)}"
            logger.error(error_msg)
            self.history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def _enhance_input_with_gemini(self, message):
        """Sử dụng Gemini để phân tích thông minh đầu vào."""
        try:
            # Tạo prompt cho Gemini
            prompt = f"""
            Bạn là trợ lý AI phân tích tin nhắn cho hệ thống đề xuất phim.

            Hãy phân tích tin nhắn sau của người dùng: "{message}"

            Phân tích chi tiết:
            1. Xác định ngữ cảnh (movie hoặc vip)
            2. Nhận diện ý định cụ thể (đề xuất, tìm kiếm, thích, không thích, v.v.)
            3. Trích xuất các thực thể quan trọng (tên phim, thể loại, loại phim, v.v.)
            4. Xử lý tiếng Việt không dấu
            5. Phân tích tâm trạng người dùng
            
            Gợi ý:
            - Nếu người dùng đề cập đến "phim hanh dong" nghĩa là "phim hành động"
            - Nếu người dùng đề cập đến "phim kinh di" nghĩa là "phim kinh dị"
            - Nếu người dùng đề cập đến "phim tinh cam" nghĩa là "phim tình cảm"
            - Nếu người dùng nói về "vip", "gói", "thanh toán", đó là ngữ cảnh vip

            Trả về dưới dạng JSON:
            {{
                "context_type": "movie/vip", 
                "intent": "recommend/search/like/dislike/info/payment/etc",
                "entities": {{
                    "movie_title": "tên phim hoặc null nếu không có",
                    "genres": ["thể loại 1", "thể loại 2"] hoặc [],
                    "film_type": "loại phim hoặc null",
                    "vip_package": "gói vip hoặc null"
                }},
                "normalized_message": "chuỗi đã chuẩn hóa",
                "user_mood": "tích cực/tiêu cực/trung tính",
                "confidence": 0.0-1.0
            }}
            """
            
            # Tối ưu hóa tốc độ bằng cách sử dụng mô hình nhẹ hơn
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            # Trích xuất JSON từ phản hồi
            enhanced_analysis = self._extract_json_from_text(response.text)
            
            # Fallback nếu xử lý thất bại
            if not enhanced_analysis:
                return self._analyze_intent(message)
            
            return enhanced_analysis
        except Exception as e:
            logger.error(f"Lỗi khi phân tích đầu vào với Gemini: {str(e)}")
            # Fallback về phương pháp phân tích ý định hiện tại
            return self._analyze_intent(message)
    
    def _handle_enhanced_movie_context(self, message, analysis):
        """Xử lý ngữ cảnh phim với phân tích nâng cao từ Gemini."""
        intent = analysis.get("intent", "general")
        entities = analysis.get("entities", {})
        
        # Xử lý theo ý định
        if intent in ["recommend", "suggestion"]:
            return self._generate_enhanced_recommendations(entities)
        elif intent in ["search", "find"]:
            return self._handle_search(message, entities)
        elif intent == "like":
            return self._handle_like(entities)
        elif intent == "dislike":
            return self._handle_dislike(entities)
        else:
            # Chat thông thường
            return self._generate_movie_chat_response(message)
    
    def _generate_enhanced_recommendations(self, entities):
        """Tạo đề xuất phim nâng cao với khả năng giới thiệu phim ngẫu nhiên."""
        movie_title = entities.get("movie_title")
        genres = entities.get("genres", [])
        film_type = entities.get("film_type")
        
        # Cập nhật sở thích người dùng
        self._update_preferences_from_entities(entities)
        
        # Ưu tiên các phương pháp đề xuất 
        recommendations = []
        source_text = ""
        
        if movie_title:
            recommendations = self.recommender.get_recommendations(movie_title=movie_title)
            source_text = f"phim \"{movie_title}\""
        elif genres:
            recommendations = self.recommender.get_genre_recommendations(genres, film_type)
            genres_text = ", ".join(genres)
            film_type_text = f" ({film_type})" if film_type else ""
            source_text = f"thể loại {genres_text}{film_type_text}"
        elif film_type:
            recommendations = self.recommender.get_film_type_recommendations(film_type)
            source_text = f"{film_type}"
        else:
            # Sử dụng sở thích người dùng
            if self.user_preferences["genres"]:
                recommendations = self.recommender.get_genre_recommendations(self.user_preferences["genres"])
                source_text = f"sở thích của bạn"
            elif self.user_preferences["liked_movies"]:
                movie_title = self.user_preferences["liked_movies"][0]
                recommendations = self.recommender.get_recommendations(movie_title=movie_title)
                source_text = f"phim bạn đã thích trước đây"
        
        # Nếu không tìm thấy đề xuất phù hợp, giới thiệu phim ngẫu nhiên
        if not recommendations:
            logger.info(f"Không tìm thấy phim phù hợp với {source_text}, chuyển sang giới thiệu phim ngẫu nhiên")
            return self._recommend_random_movies(source_text)
        
        # Format đề xuất theo định dạng Markdown
        return self._format_recommendations_to_markdown(recommendations, source_text)
    
    def _recommend_random_movies(self, source_text, count=3):
        """Giới thiệu phim ngẫu nhiên khi không tìm thấy phim phù hợp với tiêu chí."""
        try:
            # Chọn ngẫu nhiên các phim từ dataframe
            if len(self.recommender.df) <= count:
                random_movies = self.recommender.df
            else:
                random_movies = self.recommender.df.sample(n=count)
            
            # Chuyển các dòng thành danh sách từ điển
            recommendations = []
            for _, row in random_movies.iterrows():
                movie_data = row.to_dict()
                # Loại bỏ combined_features khỏi đầu ra
                if 'combined_features' in movie_data:
                    del movie_data['combined_features']
                recommendations.append(movie_data)
            
            # Tạo prompt để Gemini giới thiệu phim ngẫu nhiên
            movie_titles = [movie.get('title', 'Unknown') for movie in recommendations]
            movie_genres = [movie.get('genres', 'Unknown') for movie in recommendations]
            intro_prompt = f"""
            Người dùng đang tìm kiếm phim liên quan đến "{source_text}" nhưng không tìm thấy phim phù hợp.
            
            Hãy giới thiệu những phim ngẫu nhiên sau đây một cách hấp dẫn:
            {', '.join([f'{title} ({genre})' for title, genre in zip(movie_titles, movie_genres)])}
            
            Viết một đoạn văn ngắn (2-3 câu) giải thích rằng không có phim chính xác phù hợp với yêu cầu, 
            nhưng người dùng có thể quan tâm đến những phim thú vị này. Giọng điệu thân thiện và gợi mở.
            """
            
            try:
                intro_response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=intro_prompt
                )
                introduction = intro_response.text.strip()
            except Exception as e:
                logger.error(f"Lỗi khi tạo giới thiệu với Gemini: {str(e)}")
                introduction = f"Tôi không tìm thấy phim nào phù hợp với {source_text}, nhưng đây là một số phim thú vị khác mà bạn có thể quan tâm:"
            
            # Format đề xuất với phần giới thiệu được tạo bởi Gemini
            response = f"# Phim Đề Xuất Cho Bạn\n\n{introduction}\n\n"
            
            # Thêm thông tin chi tiết về từng phim
            for movie in recommendations:
                title = movie.get('title', 'Không có tiêu đề')
                response += f"## {title}\n\n"
                
                # Thông tin phát hành
                if movie.get('year'):
                    response += f"* **Năm phát hành**: {movie.get('year')}\n"
                
                # Thể loại và loại phim
                genre_info = []
                if movie.get('genres'):
                    genre_info.append(f"**Thể loại**: {movie.get('genres')}")
                if movie.get('film_type'):
                    genre_info.append(f"**Loại phim**: {movie.get('film_type')}")
                if genre_info:
                    response += f"* {' | '.join(genre_info)}\n"
                
                # Đạo diễn
                if movie.get('director'):
                    response += f"* **Đạo diễn**: {movie.get('director')}\n"
                
                # Tóm tắt nội dung
                if movie.get('plot'):
                    response += f"* **Tóm tắt phim**: {movie.get('plot')}\n"
                
                # Đường dẫn
                if movie.get('slug'):
                    response += f"* **Đường dẫn**: http://localhost:5173/{movie.get('slug')}\n\n"
                else:
                    safe_title = title.replace(' ', '-').lower()
                    response += f"* **Đường dẫn**: http://localhost:5173/{safe_title}\n\n"
                
                # Hình ảnh
                image_url = movie.get('poster_url', f"https://via.placeholder.com/500x300?text={title.replace(' ', '+')}")
                response += f"![Hình ảnh phim {title}]({image_url})\n\n"
                
                response += "---\n\n"
            
            # Thêm câu hỏi theo dõi
            response += "Bạn có thấy hứng thú với bất kỳ bộ phim nào trong số này không? Hoặc bạn muốn tìm kiếm với tiêu chí khác?"
            
            return response
            
        except Exception as e:
            logger.error(f"Lỗi khi đề xuất phim ngẫu nhiên: {str(e)}")
            return "Xin lỗi, tôi không tìm thấy phim phù hợp với yêu cầu của bạn. Bạn có thể thử với tiêu chí khác được không?"
    
    def _enhance_output_with_gemini(self, response_text):
        """Sử dụng Gemini để làm cho phản hồi tự nhiên và thân thiện hơn."""
        try:
            # Kiểm tra nếu phản hồi đã ở dạng Markdown với định dạng phim
            if '# ' in response_text and '## ' in response_text and '![' in response_text:
                # Đã là định dạng markdown đẹp, không cần xử lý thêm
                return response_text
            
            # Chỉ xử lý với văn bản thông thường
            prompt = f"""
            Nhiệm vụ: Cải thiện phản hồi của chatbot phim để trở nên tự nhiên và hấp dẫn hơn.

            Phản hồi gốc:
            "{response_text}"

            Yêu cầu:
            1. Làm cho văn bản trở nên thân thiện và tự nhiên hơn
            2. Thêm chút cá tính và sự nhiệt tình về phim
            3. Đảm bảo giữ nguyên thông tin và ý nghĩa
            4. Sắp xếp thông tin một cách rõ ràng và hấp dẫn
            5. Sử dụng tiếng Việt tự nhiên

            Trả lời dưới dạng đoạn văn hoàn chỉnh, không cần tiêu đề hay chú thích.
            """
            
            # Lựa chọn mô hình nhanh hơn cho xử lý đầu ra
            output_response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            enhanced_response = output_response.text.strip()
            
            # Nếu Gemini không trả về gì có ý nghĩa, giữ nguyên phản hồi gốc
            if not enhanced_response or len(enhanced_response) < len(response_text) / 2:
                return response_text
                
            return enhanced_response
        except Exception as e:
            logger.error(f"Lỗi khi xử lý đầu ra với Gemini: {str(e)}")
            # Trả về phản hồi gốc nếu có lỗi
            return response_text
    
    def _update_preferences_from_entities(self, entities):
        """Cập nhật sở thích người dùng từ các thực thể được trích xuất."""
        # Cập nhật thể loại
        if entities.get("genres"):
            for genre in entities["genres"]:
                if genre and genre not in self.user_preferences["genres"]:
                    self.user_preferences["genres"].append(genre)
        
        # Cập nhật loại phim
        if entities.get("film_type") and entities["film_type"] not in self.user_preferences["film_types"]:
            self.user_preferences["film_types"].append(entities["film_type"])
        
        # Cập nhật phim đã đề cập
        if entities.get("movie_title") and entities["movie_title"] not in self.user_preferences["mentioned_movies"]:
            self.user_preferences["mentioned_movies"].append(entities["movie_title"])
    def reset_user_data(self, reset_history=True, reset_preferences=True):
        if reset_history:
            self.history = []
            logger.info("Đã xóa lịch sử trò chuyện của người dùng")
        
        if reset_preferences:
            # Đặt lại sở thích người dùng về trạng thái ban đầu
            self.user_preferences = {
                "genres": [],
                "film_types": [],
                "liked_movies": [],
                "disliked_movies": [],
                "mentioned_movies": []
            }
            logger.info("Đã đặt lại sở thích người dùng")
        
        return {
            "success": True,
            "message": "Dữ liệu người dùng đã được đặt lại",
            "reset_history": reset_history,
            "reset_preferences": reset_preferences
        }
    def _analyze_intent(self, message):
        """Phân tích ý định của tin nhắn người dùng sử dụng Gemini."""
        try:
            # Xử lý trực tiếp một số trường hợp đặc biệt
            message_lower = message.lower()
            
            # Kiểm tra tiếng Việt không dấu
            for non_accent, with_accent in VIETNAMESE_MAPPING.items():
                if non_accent in message_lower:
                    message = message.replace(non_accent, with_accent)
                    break
            
            # Kiểm tra các từ khóa VIP
            vip_keywords = ["vip", "gói", "đăng ký", "subscription", "thanh toán", "trả góp", "payment"]
            movie_keywords = ["phim", "film", "movie", "xem", "đề xuất", "thể loại", "recommend"]
            
            # Xác định cơ bản về intent_type
            if any(keyword in message_lower for keyword in vip_keywords):
                intent_type = "vip"
            elif any(keyword in message_lower for keyword in movie_keywords):
                intent_type = "movie"
            else:
                intent_type = "chat"
            
            # Sử dụng Gemini cho phân tích chi tiết
            prompt = f"""
            Phân tích ngắn gọn ý định và các thực thể quan trọng trong tin nhắn sau:
            
            "{message}"
            
            Dựa trên phân tích, hãy trả về thông tin dưới dạng JSON với cấu trúc:
            {{
              "intent_type": "movie/vip/chat",
              "specific_intent": "recommend/search/like/dislike/info/subscribe/payment/general",
              "entities": {{
                "movie_title": "tên phim nếu có hoặc null",
                "genres": ["thể loại 1", "thể loại 2"],
                "film_type": "loại phim nếu có hoặc null",
                "vip_package": "gói VIP nếu có hoặc null"
              }},
              "user_sentiment": "positive/negative/neutral",
              "expected_response": "Mô tả ngắn về loại câu trả lời mong đợi"
            }}
            
            Chỉ trả về JSON, không thêm giải thích.
            """
            
            # Gọi Gemini API
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            # Trích xuất JSON từ phản hồi
            json_result = self._extract_json_from_text(response.text)
            
            # Nếu không thể trích xuất JSON, trả về kết quả cơ bản
            if not json_result:
                return {
                    "context_type": intent_type,
                    "intent": "general",
                    "entities": {
                        "movie_title": None,
                        "genres": [],
                        "film_type": None,
                        "vip_package": None
                    }
                }
            
            # Chuyển đổi kết quả phân tích để phù hợp với định dạng mong muốn
            result = {
                "context_type": json_result.get("intent_type", intent_type),
                "intent": json_result.get("specific_intent", "general"),
                "entities": json_result.get("entities", {})
            }
            
            return result
        except Exception as e:
            logger.error(f"Lỗi khi phân tích ý định: {str(e)}")
            return {
                "context_type": "movie",
                "intent": "general",
                "entities": {
                    "movie_title": None,
                    "genres": [],
                    "film_type": None
                }
            }
    
    def _handle_vip_context(self, message, intent_analysis):
        """Xử lý tin nhắn trong ngữ cảnh VIP."""
        # Tạo prompt cho Gemini với thông tin VIP
        recent_history = self._get_recent_conversation_history(5)
        
        prompt = f"""
        {self.vip_system_prompt}
        
        Lịch sử cuộc trò chuyện gần đây:
        {recent_history}
        
        Tin nhắn của người dùng: "{message}"
        
        Hãy trả lời bằng định dạng Markdown.
        """
        
        # Gọi Gemini API
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return response.text
    
    def _handle_movie_context(self, message, intent_analysis):
        """Xử lý tin nhắn trong ngữ cảnh phim."""
        specific_intent = intent_analysis.get("specific_intent", "general")
        entities = intent_analysis.get("entities", {})
        
        # Kiểm tra từng intent cụ thể
        if specific_intent == "recommend":
            return self._generate_recommendations(entities)
        elif specific_intent == "search":
            return self._handle_search(message, entities)
        elif specific_intent == "like":
            return self._handle_like(entities)
        elif specific_intent == "dislike":
            return self._handle_dislike(entities)
        else:
            # Chat thông thường với Gemini
            return self._generate_movie_chat_response(message)
    
    def _generate_recommendations(self, entities):
        """Tạo đề xuất phim dựa trên phân tích ý định."""
        movie_title = entities.get("movie_title")
        genres = entities.get("genres", [])
        film_type = entities.get("film_type")
        
        # Cập nhật sở thích
        for genre in genres:
            if genre and genre not in self.user_preferences["genres"]:
                self.user_preferences["genres"].append(genre)
        
        if film_type and film_type not in self.user_preferences["film_types"]:
            self.user_preferences["film_types"].append(film_type)
        
        # Lấy đề xuất từ recommender
        if movie_title:
            recommendations = self.recommender.get_recommendations(movie_title=movie_title)
            source_text = f"phim \"{movie_title}\""
        elif genres:
            recommendations = self.recommender.get_genre_recommendations(genres, film_type)
            genres_text = ", ".join(genres)
            film_type_text = f" thuộc {film_type}" if film_type else ""
            source_text = f"thể loại {genres_text}{film_type_text}"
        elif film_type:
            recommendations = self.recommender.get_film_type_recommendations(film_type)
            source_text = f"{film_type}"
        else:
            # Sử dụng sở thích người dùng
            if self.user_preferences["genres"]:
                recommendations = self.recommender.get_genre_recommendations(self.user_preferences["genres"])
                source_text = f"sở thích của bạn (thể loại {', '.join(self.user_preferences['genres'])})"
            elif self.user_preferences["liked_movies"]:
                movie_title = self.user_preferences["liked_movies"][0]
                recommendations = self.recommender.get_recommendations(movie_title=movie_title)
                source_text = f"phim bạn đã thích (\"{movie_title}\")"
            else:
                recommendations = self.recommender.get_popular_movies()
                source_text = "các phim phổ biến"
        
        # Tạo câu trả lời Markdown
        if not recommendations:
            # Nếu không tìm thấy, chuyển sang giới thiệu phim ngẫu nhiên
            logger.info(f"Không tìm thấy phim phù hợp với {source_text}, chuyển sang giới thiệu phim ngẫu nhiên")
            return self._recommend_random_movies(source_text)
        
        # Format đề xuất theo định dạng Markdown
        return self._format_recommendations_to_markdown(recommendations, source_text)
    
    def _handle_search(self, message, entities):
        """Xử lý yêu cầu tìm kiếm phim."""
        # Trích xuất từ khóa tìm kiếm từ tin nhắn
        search_query = message.lower()
        for term in ["tìm", "kiếm", "search", "find", "looking for", "tìm kiếm"]:
            search_query = search_query.replace(term, "")
        
        search_query = search_query.strip()
        
        if not search_query and entities.get("movie_title"):
            search_query = entities["movie_title"]
        
        if not search_query:
            return "Bạn muốn tìm kiếm phim gì? Hãy cho tôi biết tên phim, diễn viên, đạo diễn hoặc thể loại."
        
        # Thực hiện tìm kiếm
        results = self.recommender.search_movies(search_query)
        
        if not results:
            # Nếu không tìm thấy, chuyển sang giới thiệu phim ngẫu nhiên
            logger.info(f"Không tìm thấy kết quả cho tìm kiếm: {search_query}, chuyển sang giới thiệu phim ngẫu nhiên")
            return self._recommend_random_movies(search_query)
        
        # Format kết quả tìm kiếm
        response = f"# Kết Quả Tìm Kiếm: '{search_query}'\n\n"
        
        for movie in results:
            title = movie.get('title', 'Không có tiêu đề')
            response += f"## {title}\n\n"
            
            # Thêm thông tin phim
            if movie.get('year'):
                response += f"* **Năm phát hành**: {movie.get('year')}\n"
            
            genre_info = []
            if movie.get('genres'):
                genre_info.append(f"**Thể loại**: {movie.get('genres')}")
            if movie.get('film_type'):
                genre_info.append(f"**Loại phim**: {movie.get('film_type')}")
            if genre_info:
                response += f"* {' | '.join(genre_info)}\n"
            
            if movie.get('director'):
                response += f"* **Đạo diễn**: {movie.get('director')}\n"
                
            if movie.get('plot'):
                response += f"* **Tóm tắt phim**: {movie.get('plot')}\n"
            
            # Thêm đường dẫn
            if movie.get('slug'):
                response += f"* **Đường dẫn**: http://localhost:5173/{movie.get('slug')}\n\n"
            else:
                safe_title = title.replace(' ', '-').lower()
                response += f"* **Đường dẫn**: http://localhost:5173/{safe_title}\n\n"
            
            # Thêm hình ảnh
            image_url = movie.get('poster_url', f"https://via.placeholder.com/500x300?text={title.replace(' ', '+')}")
            response += f"![Hình ảnh phim {title}]({image_url})\n\n"
            
            response += "---\n\n"
        
        response += "Bạn có quan tâm đến phim nào trong số này không? Tôi có thể giúp bạn tìm hiểu thêm về bất kỳ bộ phim nào."
        
        return response
    
    def _handle_like(self, entities):
        """Xử lý khi người dùng thích một bộ phim."""
        movie_title = entities.get("movie_title")
        
        if not movie_title:
            return "Tôi rất vui khi bạn thích! Bạn có thể cho tôi biết tên phim cụ thể mà bạn đã thích không? Điều đó sẽ giúp tôi đề xuất các phim tương tự hơn."
        
        # Thêm vào danh sách phim đã thích
        if movie_title not in self.user_preferences["liked_movies"]:
            self.user_preferences["liked_movies"].append(movie_title)
        
        # Trích xuất thể loại từ phim để cập nhật sở thích
        idx = self.recommender.get_movie_index(movie_title)
        if idx is not None:
            try:
                movie_row = self.recommender.df.iloc[idx]
                if 'genres' in movie_row and movie_row['genres']:
                    genres = str(movie_row['genres']).split('|')
                    for genre in genres:
                        genre = genre.strip()
                        if genre and genre not in self.user_preferences["genres"]:
                            self.user_preferences["genres"].append(genre)
                
                if 'film_type' in movie_row and movie_row['film_type']:
                    film_type = movie_row['film_type']
                    if film_type and film_type not in self.user_preferences["film_types"]:
                        self.user_preferences["film_types"].append(film_type)
            except Exception as e:
                logger.error(f"Lỗi khi cập nhật sở thích từ phim: {str(e)}")
        
        # Tạo phản hồi đề xuất thêm phim tương tự
        try:
            # Lấy phim tương tự
            similar_movies = self.recommender.get_recommendations(movie_title=movie_title, top_n=3)
            
            if not similar_movies:
                message = f"Tôi rất vui khi bạn thích phim \"{movie_title}\"! Tôi đã ghi nhận sở thích này và sẽ sử dụng thông tin này để đưa ra đề xuất phù hợp hơn trong tương lai."
                return self._generate_movie_chat_response(message)
            
            # Tạo phản hồi
            liked_prompt = f"""
            Người dùng vừa cho biết họ thích phim "{movie_title}". Hãy viết một đoạn phản hồi thân thiện, 
            nói về phim đó (nếu bạn biết) và đề xuất 3 phim tương tự dưới đây. Đảm bảo đề cập đặc điểm tương đồng.
            
            Các phim tương tự:
            """
            
            # Thêm thông tin về 3 phim tương tự
            for movie in similar_movies:
                title = movie.get('title', 'Không có tiêu đề')
                genre = movie.get('genres', 'Không có thể loại')
                plot = movie.get('plot', 'Không có mô tả')
                
                liked_prompt += f"- {title}: {genre}. {plot}\n"
            
            # Gọi Gemini để có phản hồi tự nhiên hơn
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=liked_prompt
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Lỗi khi tạo phản hồi thích phim: {str(e)}")
            return f"Tôi rất vui khi bạn thích phim \"{movie_title}\"! Tôi sẽ ghi nhớ điều này để đề xuất các phim tương tự cho bạn trong tương lai."
    
    def _handle_dislike(self, entities):
        """Xử lý khi người dùng không thích một bộ phim."""
        movie_title = entities.get("movie_title")
        
        if not movie_title:
            return "Tôi hiểu rằng bạn không thích phim đó. Bạn có thể cho tôi biết tên phim cụ thể để tôi có thể tránh đề xuất các phim tương tự trong tương lai không?"
        
        # Thêm vào danh sách phim không thích
        if movie_title not in self.user_preferences["disliked_movies"]:
            self.user_preferences["disliked_movies"].append(movie_title)
        
        # Xóa khỏi danh sách phim đã thích nếu có
        if movie_title in self.user_preferences["liked_movies"]:
            self.user_preferences["liked_movies"].remove(movie_title)
        
        dislike_prompt = f"""
        Người dùng vừa nói họ không thích phim "{movie_title}". 
        
        Hãy viết một phản hồi thân thiện, thể hiện sự thấu hiểu, và đề xuất họ thử một phim hoàn toàn khác biệt.
        Đề xuất phim thuộc thể loại khác hoặc phong cách khác, không phải phim tương tự.
        
        Kết thúc bằng câu hỏi về loại phim họ thực sự thích.
        """
        
        # Gọi Gemini để có phản hồi tự nhiên
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=dislike_prompt
        )
        
        return response.text
    
    def _generate_movie_chat_response(self, message):
        """Tạo phản hồi chat từ Gemini dựa trên lịch sử và sở thích."""
        # Lấy lịch sử trò chuyện gần đây
        recent_history = self._get_recent_conversation_history(5)
        
        # Tạo thông tin về sở thích người dùng
        preferences_info = ""
        if self.user_preferences["genres"]:
            preferences_info += f"Thể loại yêu thích: {', '.join(self.user_preferences['genres'])}\n"
        if self.user_preferences["film_types"]:
            preferences_info += f"Loại phim yêu thích: {', '.join(self.user_preferences['film_types'])}\n"
        if self.user_preferences["liked_movies"]:
            preferences_info += f"Phim đã thích: {', '.join(self.user_preferences['liked_movies'])}\n"
        if self.user_preferences["disliked_movies"]:
            preferences_info += f"Phim không thích: {', '.join(self.user_preferences['disliked_movies'])}\n"
        
        # Tạo prompt cho Gemini
        prompt = f"""
        {self.movie_system_prompt}
        
        Thông tin về sở thích người dùng:
        {preferences_info}
        
        Lịch sử trò chuyện gần đây:
        {recent_history}
        
        Tin nhắn hiện tại của người dùng: "{message}"
        
        Hãy trả lời chi tiết, hữu ích và tự nhiên. Nếu thích hợp, đề xuất phim dựa trên sở thích của người dùng.
        """
        
        # Gọi Gemini API
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return response.text
    
    def _format_recommendations_to_markdown(self, recommendations, source_text):
        """Format các đề xuất phim thành định dạng Markdown đẹp."""
        # Tạo prompt cho Gemini để có giới thiệu tự nhiên
        intro_prompt = f"""
        Viết một đoạn văn ngắn (2-3 câu) giới thiệu về đề xuất phim dựa trên {source_text}. 
        Đoạn văn nên thân thiện và hấp dẫn, không quá dài dòng.
        """
        
        try:
            intro_response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=intro_prompt
            )
            introduction = intro_response.text.strip()
        except Exception as e:
            logger.error(f"Lỗi khi tạo giới thiệu: {str(e)}")
            introduction = f"Dựa trên {source_text}, tôi đã tìm ra một số phim mà bạn có thể sẽ thích:"
        
        # Tạo phần đầu của phản hồi
        response = f"# Đề Xuất Phim Dựa Trên {source_text}\n\n{introduction}\n\n"
        
        # Thêm mỗi phim vào phản hồi
        for movie in recommendations:
            title = movie.get('title', 'Không có tiêu đề')
            response += f"## {title}\n\n"
            
            # Thông tin phát hành
            if movie.get('year'):
                response += f"* **Năm phát hành**: {movie.get('year')}\n"
            else:
                response += f"* **Năm phát hành**: Không có thông tin\n"
            
            # Thể loại và loại phim
            genre_info = []
            if movie.get('genres'):
                genre_info.append(f"**Thể loại**: {movie.get('genres')}")
            if movie.get('film_type'):
                genre_info.append(f"**Loại phim**: {movie.get('film_type')}")
            if genre_info:
                response += f"* {' | '.join(genre_info)}\n"
            
            # Đạo diễn
            if movie.get('director'):
                response += f"* **Đạo diễn**: {movie.get('director')}\n"
            
            # Tóm tắt nội dung
            if movie.get('plot'):
                response += f"* **Tóm tắt phim**: {movie.get('plot')}\n"
            else:
                response += f"* **Tóm tắt phim**: Thông tin đang được cập nhật.\n"
            
            # Đường dẫn
            if movie.get('slug'):
                response += f"* **Đường dẫn**: http://localhost:5173/{movie.get('slug')}\n\n"
            else:
                safe_title = title.replace(' ', '-').lower()
                response += f"* **Đường dẫn**: http://localhost:5173/{safe_title}\n\n"
            
            # Hình ảnh
            image_url = movie.get('poster_url', f"https://via.placeholder.com/500x300?text={title.replace(' ', '+')}")
            response += f"![Hình ảnh phim {title}]({image_url})\n\n"
            
            response += "---\n\n"
        
        # Thêm câu hỏi theo dõi
        follow_up_prompt = f"""
        Viết một câu hỏi theo dõi ngắn để hỏi người dùng liệu họ có hứng thú với các đề xuất phim 
        dựa trên {source_text} không. Câu hỏi nên ngắn gọn và tự nhiên.
        """
        
        try:
            follow_up_response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=follow_up_prompt
            )
            follow_up = follow_up_response.text.strip()
        except Exception as e:
            logger.error(f"Lỗi khi tạo câu hỏi theo dõi: {str(e)}")
            follow_up = "Bạn có thấy hứng thú với bất kỳ bộ phim nào trong số này không?"
        
        response += follow_up
        
        return response
    
    def _get_recent_conversation_history(self, max_entries=5):
        """Lấy lịch sử trò chuyện gần đây dưới dạng văn bản."""
        if not self.history:
            return "Chưa có lịch sử trò chuyện."
        
        recent_history = self.history[-min(len(self.history), max_entries*2):]
        formatted_history = "\n".join([f"{entry['role']}: {entry['content']}" for entry in recent_history])
        
        return formatted_history
    
    def _extract_json_from_text(self, text):
        """Trích xuất dữ liệu JSON từ văn bản."""
        try:
            # Tìm chuỗi JSON trong văn bản
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result
            return None
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất JSON: {str(e)}")
            return None



def convert_vietnamese_no_accent(text):
    """
    Chuyển đổi tiếng Việt không dấu sang có dấu cho các từ thông dụng.
    """
    # Kiểm tra text hoàn chỉnh
    text_lower = text.lower()
    if text_lower in VIETNAMESE_MAPPING:
        return VIETNAMESE_MAPPING[text_lower]
    
    # Kiểm tra từng từ trong câu
    words = text_lower.split()
    result = []
    
    i = 0
    while i < len(words):
        # Kiểm tra 2-gram (cặp từ)
        if i < len(words) - 1:
            bigram = words[i] + " " + words[i+1]
            if bigram in VIETNAMESE_MAPPING:
                result.append(VIETNAMESE_MAPPING[bigram])
                i += 2
                continue
                
        # Kiểm tra 1-gram (từ đơn)
        if words[i] in VIETNAMESE_MAPPING:
            result.append(VIETNAMESE_MAPPING[words[i]])
        else:
            result.append(words[i])
        i += 1
    
    return " ".join(result)
class UserPreferenceManager:
    """Quản lý sở thích người dùng với cơ sở dữ liệu MySQL."""
    
    def __init__(self, db_config: Dict[str, Any], pool_size: int = 5):
        """Khởi tạo kết nối đến cơ sở dữ liệu."""
        self.connection_pool = pooling.MySQLConnectionPool(
            pool_name="preference_pool",
            pool_size=pool_size,
            **db_config
        )
    
    def get_connection(self):
        """Lấy kết nối từ pool."""
        return self.connection_pool.get_connection()
    
    def create_session(self, ip_address: str, user_agent: str, 
                    user_id: Optional[int] = None) -> str:
        """Tạo phiên mới và trả về session_id."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            session_id = str(uuid.uuid4())
            expires_at = datetime.datetime.now() + datetime.timedelta(days=30)
            
            query = """
            INSERT INTO sessions (id, user_id, ip_address, user_agent, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (session_id, user_id, ip_address, user_agent, expires_at))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return session_id
        except Exception as e:
            print(f"Error creating session: {e}")
            raise
    
    def get_user_preferences(self, user_id: Optional[int] = None, 
                        session_id: Optional[str] = None) -> Dict[str, Any]:
        """Lấy tất cả sở thích của người dùng."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Kiểm tra xem có bản tóm tắt sở thích không
            summary_query = """
            SELECT preference_data FROM user_preference_summary
            WHERE user_id = %s OR session_id = %s
            ORDER BY updated_at DESC LIMIT 1
            """
            cursor.execute(summary_query, (user_id, session_id))
            summary = cursor.fetchone()
            
            if summary:
                preferences = json.loads(summary['preference_data'])
            else:
                # Lấy sở thích từ các bảng riêng lẻ
                preferences = {
                    "genres": self._get_genre_preferences(cursor, user_id, session_id),
                    "film_types": self._get_film_type_preferences(cursor, user_id, session_id),
                    "liked_movies": self._get_movie_interactions(cursor, user_id, session_id, "like"),
                    "disliked_movies": self._get_movie_interactions(cursor, user_id, session_id, "dislike"),
                    "viewed_movies": self._get_movie_interactions(cursor, user_id, session_id, "view"),
                    "search_history": self._get_search_history(cursor, user_id, session_id)
                }
                
                # Lưu tóm tắt để truy vấn nhanh hơn trong tương lai
                self._save_preference_summary(cursor, user_id, session_id, preferences)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return preferences
        except Exception as e:
            print(f"Error getting user preferences: {e}")
            return {
                "genres": [],
                "film_types": [],
                "liked_movies": [],
                "disliked_movies": [],
                "viewed_movies": [],
                "search_history": []
            }
    def remove_genre_preference(self, genre, user_id=None, session_id=None):
        """Xóa thể loại yêu thích."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Xóa thể loại
            delete_query = """
            DELETE FROM user_genre_preferences
            WHERE (user_id = %s OR session_id = %s) AND genre = %s
            """
            cursor.execute(delete_query, (user_id, session_id, genre))
            
            # Đánh dấu cần cập nhật tóm tắt
            self._invalidate_summary(cursor, user_id, session_id)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa thể loại yêu thích: {str(e)}")
            return False

    def remove_movie_interaction(self, movie_id, interaction_type, user_id=None, session_id=None):
        """Xóa tương tác với phim."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Xóa tương tác
            delete_query = """
            DELETE FROM user_movie_interactions
            WHERE (user_id = %s OR session_id = %s) AND movie_id = %s AND interaction_type = %s
            """
            cursor.execute(delete_query, (user_id, session_id, movie_id, interaction_type))
            
            # Đánh dấu cần cập nhật tóm tắt
            self._invalidate_summary(cursor, user_id, session_id)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa tương tác với phim: {str(e)}")
            return False

    def _get_genre_preferences(self, cursor, user_id, session_id):
        """Lấy thể loại yêu thích."""
        query = """
        SELECT genre, weight FROM user_genre_preferences
        WHERE (user_id = %s OR session_id = %s)
        ORDER BY weight DESC
        """
        cursor.execute(query, (user_id, session_id))
        return [{"genre": row['genre'], "weight": row['weight']} for row in cursor.fetchall()]
    
    def _get_film_type_preferences(self, cursor, user_id, session_id):
        """Lấy loại phim yêu thích."""
        query = """
        SELECT film_type, weight FROM user_film_type_preferences
        WHERE (user_id = %s OR session_id = %s)
        ORDER BY weight DESC
        """
        cursor.execute(query, (user_id, session_id))
        return [{"film_type": row['film_type'], "weight": row['weight']} for row in cursor.fetchall()]
    
    def _get_movie_interactions(self, cursor, user_id, session_id, interaction_type):
        """Lấy danh sách phim đã tương tác."""
        query = """
        SELECT movie_id, created_at FROM user_movie_interactions
        WHERE (user_id = %s OR session_id = %s) AND interaction_type = %s
        ORDER BY created_at DESC
        """
        cursor.execute(query, (user_id, session_id, interaction_type))
        return [row['movie_id'] for row in cursor.fetchall()]
    
    def _get_search_history(self, cursor, user_id, session_id):
        """Lấy lịch sử tìm kiếm."""
        query = """
        SELECT query, created_at FROM search_history
        WHERE (user_id = %s OR session_id = %s)
        ORDER BY created_at DESC LIMIT 20
        """
        cursor.execute(query, (user_id, session_id))
        return [row['query'] for row in cursor.fetchall()]
    
    def _save_preference_summary(self, cursor, user_id, session_id, preferences):
        """Lưu tóm tắt sở thích để truy vấn nhanh."""
        preference_json = json.dumps(preferences)
        
        query = """
        INSERT INTO user_preference_summary (user_id, session_id, preference_data)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE preference_data = %s, updated_at = CURRENT_TIMESTAMP
        """
        cursor.execute(query, (user_id, session_id, preference_json, preference_json))
    
    def add_genre_preference(self, genre: str, user_id: Optional[int] = None, 
                        session_id: Optional[str] = None, weight: float = 1.0):
        """Thêm hoặc cập nhật thể loại yêu thích."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Kiểm tra xem đã có chưa
            check_query = """
            SELECT id, weight FROM user_genre_preferences
            WHERE (user_id = %s OR session_id = %s) AND genre = %s
            """
            cursor.execute(check_query, (user_id, session_id, genre))
            existing = cursor.fetchone()
            
            if existing:
                # Cập nhật trọng số
                new_weight = min(5.0, existing[1] + weight)  # Giới hạn tối đa là 5.0
                update_query = """
                UPDATE user_genre_preferences SET weight = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                cursor.execute(update_query, (new_weight, existing[0]))
            else:
                # Thêm mới
                insert_query = """
                INSERT INTO user_genre_preferences (user_id, session_id, genre, weight)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_query, (user_id, session_id, genre, weight))
            
            # Đánh dấu cần cập nhật tóm tắt
            self._invalidate_summary(cursor, user_id, session_id)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error adding genre preference: {e}")
            return False
    
    def add_film_type_preference(self, film_type: str, user_id: Optional[int] = None, 
                            session_id: Optional[str] = None, weight: float = 1.0):
        """Thêm hoặc cập nhật loại phim yêu thích."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Kiểm tra xem đã có chưa
            check_query = """
            SELECT id, weight FROM user_film_type_preferences
            WHERE (user_id = %s OR session_id = %s) AND film_type = %s
            """
            cursor.execute(check_query, (user_id, session_id, film_type))
            existing = cursor.fetchone()
            
            if existing:
                # Cập nhật trọng số
                new_weight = min(5.0, existing[1] + weight)  # Giới hạn tối đa là 5.0
                update_query = """
                UPDATE user_film_type_preferences SET weight = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                cursor.execute(update_query, (new_weight, existing[0]))
            else:
                # Thêm mới
                insert_query = """
                INSERT INTO user_film_type_preferences (user_id, session_id, film_type, weight)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_query, (user_id, session_id, film_type, weight))
            
            # Đánh dấu cần cập nhật tóm tắt
            self._invalidate_summary(cursor, user_id, session_id)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error adding film type preference: {e}")
            return False
    
    def record_movie_interaction(self, movie_id: int, interaction_type: str, 
                            user_id: Optional[int] = None, session_id: Optional[str] = None):
        """Ghi nhận tương tác với phim (like, dislike, view, bookmark)."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Nếu là dislike, xóa like nếu có
            if interaction_type == "dislike":
                delete_query = """
                DELETE FROM user_movie_interactions
                WHERE (user_id = %s OR session_id = %s) AND movie_id = %s AND interaction_type = 'like'
                """
                cursor.execute(delete_query, (user_id, session_id, movie_id))
            
            # Nếu là like, xóa dislike nếu có
            elif interaction_type == "like":
                delete_query = """
                DELETE FROM user_movie_interactions
                WHERE (user_id = %s OR session_id = %s) AND movie_id = %s AND interaction_type = 'dislike'
                """
                cursor.execute(delete_query, (user_id, session_id, movie_id))
            
            # Kiểm tra xem đã có tương tác này chưa
            check_query = """
            SELECT id FROM user_movie_interactions
            WHERE (user_id = %s OR session_id = %s) AND movie_id = %s AND interaction_type = %s
            """
            cursor.execute(check_query, (user_id, session_id, movie_id, interaction_type))
            existing = cursor.fetchone()
            
            if not existing:
                # Thêm mới
                insert_query = """
                INSERT INTO user_movie_interactions (user_id, session_id, movie_id, interaction_type)
                VALUES (%s, %s, %s, %s)
                """
                cursor.execute(insert_query, (user_id, session_id, movie_id, interaction_type))
            
            # Đánh dấu cần cập nhật tóm tắt
            self._invalidate_summary(cursor, user_id, session_id)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error recording movie interaction: {e}")
            return False
    
    def record_search(self, query: str, user_id: Optional[int] = None, 
                    session_id: Optional[str] = None):
        """Ghi nhận tìm kiếm."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Thêm vào lịch sử tìm kiếm
            insert_query = """
            INSERT INTO search_history (user_id, session_id, query)
            VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (user_id, session_id, query))
            
            # Đánh dấu cần cập nhật tóm tắt
            self._invalidate_summary(cursor, user_id, session_id)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error recording search: {e}")
            return False
    
    def record_recommendation_impression(self, movie_id: int, recommendation_type: str,
                                    user_id: Optional[int] = None, session_id: Optional[str] = None):
        """Ghi nhận việc hiển thị đề xuất phim."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Thêm vào lịch sử hiển thị
            insert_query = """
            INSERT INTO recommendation_impressions (user_id, session_id, movie_id, recommendation_type)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (user_id, session_id, movie_id, recommendation_type))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error recording recommendation impression: {e}")
            return False
    
    def _invalidate_summary(self, cursor, user_id, session_id):
        """Xóa bản tóm tắt hiện tại để tạo mới khi cần."""
        delete_query = """
        DELETE FROM user_preference_summary
        WHERE (user_id = %s OR session_id = %s)
        """
        cursor.execute(delete_query, (user_id, session_id))
    
    def reset_preferences(self, user_id: Optional[int] = None, session_id: Optional[str] = None,
                        reset_genres: bool = True, reset_film_types: bool = True,
                        reset_interactions: bool = True, reset_history: bool = True):
        """Đặt lại sở thích người dùng."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if reset_genres:
                delete_query = """
                DELETE FROM user_genre_preferences
                WHERE (user_id = %s OR session_id = %s)
                """
                cursor.execute(delete_query, (user_id, session_id))
            
            if reset_film_types:
                delete_query = """
                DELETE FROM user_film_type_preferences
                WHERE (user_id = %s OR session_id = %s)
                """
                cursor.execute(delete_query, (user_id, session_id))
            
            if reset_interactions:
                delete_query = """
                DELETE FROM user_movie_interactions
                WHERE (user_id = %s OR session_id = %s)
                """
                cursor.execute(delete_query, (user_id, session_id))
            
            if reset_history:
                delete_query = """
                DELETE FROM search_history
                WHERE (user_id = %s OR session_id = %s)
                """
                cursor.execute(delete_query, (user_id, session_id))
            
            # Xóa bản tóm tắt
            self._invalidate_summary(cursor, user_id, session_id)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error resetting preferences: {e}")
            return False
    
    def migrate_session_to_user(self, session_id: str, user_id: int):
        """Di chuyển sở thích từ phiên ẩn danh sang tài khoản người dùng."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Cập nhật tất cả bảng
            tables = [
                "user_genre_preferences",
                "user_film_type_preferences",
                "user_movie_interactions",
                "search_history",
                "recommendation_impressions"
            ]
            
            for table in tables:
                update_query = f"""
                UPDATE {table} SET user_id = %s
                WHERE session_id = %s AND user_id IS NULL
                """
                cursor.execute(update_query, (user_id, session_id))
            
            # Cập nhật phiên
            update_session_query = """
            UPDATE sessions SET user_id = %s
            WHERE id = %s
            """
            cursor.execute(update_session_query, (user_id, session_id))
            
            # Xóa bản tóm tắt
            self._invalidate_summary(cursor, user_id, session_id)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error migrating session to user: {e}")
            return False
db_config = {
    "host": os.getenv('DB_HOST', 'localhost'),
    "user": os.getenv('DB_USER', 'root'),
    "password": os.getenv('DB_PASSWORD', ''),
    "database": os.getenv('DB_NAME', 'csdl_phim')
}

preference_manager = UserPreferenceManager(db_config)


# Khởi tạo dữ liệu
csv_path = os.getenv('MOVIE_CSV_PATH', 'vietnamese_movies.csv')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Kiểm tra API key
if not google_api_key:
    logger.warning("Không tìm thấy GOOGLE_API_KEY trong biến môi trường! Vui lòng đặt giá trị này để sử dụng Gemini AI.")

# Thử chuyển đổi dữ liệu nếu cần
try:
    try_db_conversion = False  # Set to True if you want to try database conversion
    if try_db_conversion:
        data_converter = InitDataToCSV()
        csv_path = data_converter.process_and_save(csv_path, force_update=False)
        logger.info(f"Đã kiểm tra/chuyển đổi dữ liệu sang CSV: {csv_path}")
except Exception as e:
    logger.error(f"Lỗi khi chuyển đổi dữ liệu: {str(e)}")
    logger.info("Sẽ sử dụng file CSV hiện có hoặc tạo file mẫu")

# Khởi tạo chatbot
try:
    chatbot = AIMovieChatbot(csv_path, google_api_key)
    logger.info(f"Đã khởi tạo chatbot thành công với dữ liệu từ {csv_path}")
    if google_api_key:
        logger.info("Gemini AI đã được kích hoạt")
    else:
        logger.warning("Gemini AI không được kích hoạt do thiếu API key")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo chatbot: {str(e)}")
    chatbot = None
app.secret_key = os.getenv('SECRET_KEY', 'dev_secret_key')  
# Middleware để quản lý phiên
# Middleware để quản lý phiên
@app.before_request
def handle_session():
    # Kiểm tra nếu không có session_id
    if 'session_id' not in session:
        # Tạo phiên mới
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        user_id = session.get('user_id')  # None nếu chưa đăng nhập
        
        try:
            if preference_manager:
                session_id = preference_manager.create_session(ip_address, user_agent, user_id)
                session['session_id'] = session_id
                logger.info(f"Đã tạo phiên mới: {session_id}")
            else:
                # Tạo session_id giả nếu preference_manager không khả dụng
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id
                logger.warning(f"Đã tạo phiên giả: {session_id} (preference_manager không khả dụng)")
        except Exception as e:
            logger.error(f"Lỗi khi tạo phiên: {str(e)}")
            # Tạo session_id giả trong trường hợp lỗi
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
# Định nghĩa các routes
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'success': True,
        'message': 'Chatbot đề xuất phim đang hoạt động!',
        'version': '2.0.0',
        'endpoints': {
            '/chat': 'POST - Gửi tin nhắn đến chatbot',
            '/recommendations': 'GET - Lấy đề xuất phim',
            '/search': 'GET - Tìm kiếm phim',
            '/genres': 'GET - Lấy danh sách thể loại',
            '/film-types': 'GET - Lấy danh sách loại phim'
        }
    })

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        global chatbot
        # Kiểm tra xem chatbot đã được khởi tạo chưa
        if chatbot is None:
            return jsonify({
                'success': False,
                'error': 'Chatbot chưa được khởi tạo, vui lòng thử lại sau.'
            }), 500
        
        # Xử lý dữ liệu đầu vào
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Không có dữ liệu được cung cấp'}), 400
        
        if 'message' not in data:
            return jsonify({'error': 'Thiếu trường "message" trong dữ liệu'}), 400
        
        user_message = data['message']
        context = data.get('mode', 'movie')
        
        # Lấy session_id từ phiên
        session_id = session.get('session_id')
        user_id = session.get('user_id')  # None nếu chưa đăng nhập
        
        # Lấy sở thích người dùng từ cơ sở dữ liệu
        user_preferences = preference_manager.get_user_preferences(user_id, session_id)
        
        # Cập nhật sở thích vào chatbot
        chatbot.user_preferences = {
            "genres": [p["genre"] for p in user_preferences["genres"]],
            "film_types": [p["film_type"] for p in user_preferences["film_types"]],
            "liked_movies": user_preferences["liked_movies"],
            "disliked_movies": user_preferences["disliked_movies"]
        }
        
        # Cập nhật ngữ cảnh nếu được chỉ định
        if context == 'vip':
            chatbot.conversation_context = 'vip'
        elif context == 'movie':
            chatbot.conversation_context = 'movie'
        
        # Xử lý tin nhắn
        start_time = time.time()
        response = chatbot.process_message(user_message)
        processing_time = time.time() - start_time
        
        # Cập nhật sở thích từ chatbot vào cơ sở dữ liệu
        for genre in chatbot.user_preferences["genres"]:
            if genre not in [p["genre"] for p in user_preferences["genres"]]:
                preference_manager.add_genre_preference(genre, user_id, session_id)
        
        for film_type in chatbot.user_preferences["film_types"]:
            if film_type not in [p["film_type"] for p in user_preferences["film_types"]]:
                preference_manager.add_film_type_preference(film_type, user_id, session_id)
        
        for movie_id in chatbot.user_preferences["liked_movies"]:
            if movie_id not in user_preferences["liked_movies"]:
                preference_manager.record_movie_interaction(movie_id, "like", user_id, session_id)
        
        for movie_id in chatbot.user_preferences["disliked_movies"]:
            if movie_id not in user_preferences["disliked_movies"]:
                preference_manager.record_movie_interaction(movie_id, "dislike", user_id, session_id)
        
        # Xác định loại phản hồi (markdown hoặc html)
        is_markdown = '# ' in response or '## ' in response
        
        logger.info(f"Đã xử lý tin nhắn trong {processing_time:.2f} giây, độ dài phản hồi: {len(response)}")
        
        return jsonify({
            'success': True,
            'response': response,
            'is_markdown': is_markdown,
            'history': chatbot.history,
            'context': chatbot.conversation_context,
            'preferences': chatbot.user_preferences,
            'processing_time': processing_time
        })
    except Exception as e:
        logger.error(f"Lỗi khi xử lý yêu cầu chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        global chatbot
        # Kiểm tra xem chatbot đã được khởi tạo chưa
        if chatbot is None or chatbot.recommender is None:
            return jsonify({
                'success': False,
                'error': 'Hệ thống đề xuất chưa được khởi tạo, vui lòng thử lại sau.'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Không có dữ liệu được cung cấp'
            }), 400
        
        # Trích xuất các tham số từ payload JSON
        liked_movies = data.get('liked_movies', [])
        disliked_movies = data.get('disliked_movies', [])
        genres_list = data.get('genres', [])
        film_types_list = data.get('film_types', [])
        mentioned_movies = data.get('mentioned_movies', [])
        
        # Xử lý film_type
        film_type = None
        if film_types_list and len(film_types_list) > 0:
            film_type = film_types_list[0]  # Lấy loại phim đầu tiên
            
            # Kiểm tra xem film_type có hợp lệ không
            if FILM_TYPES and film_type not in FILM_TYPES:
                # Tìm kiếm không phân biệt hoa thường
                lower_film_types = [ft.lower() for ft in FILM_TYPES]
                if film_type.lower() in lower_film_types:
                    index = lower_film_types.index(film_type.lower())
                    film_type = FILM_TYPES[index]
        
        # Xử lý genres
        genres = []
        if genres_list:
            genres = genres_list  # Sử dụng danh sách thể loại trực tiếp
        
        # Xử lý count an toàn
        count_param = request.args.get('count', '5')
        try:
            count = int(count_param) if count_param.strip() else 5
        except ValueError:
            count = 5  # Mặc định là 5 nếu không thể chuyển đổi
        
        # Giới hạn số lượng phim đề xuất
        count = max(1, min(count, 20))  # Giới hạn từ 1 đến 20 phim
        
        # Log thông tin tham số
        logger.info(f"Tham số đề xuất: liked_movies={liked_movies}, disliked_movies={disliked_movies}, " +
                   f"genres={genres}, film_type='{film_type}', mentioned_movies={mentioned_movies}, count={count}")
        
        # Lấy đề xuất
        recommendations = []
        
        # Ưu tiên theo thứ tự: phim đã đề cập > phim đã thích > thể loại > loại phim > phổ biến
        if mentioned_movies and len(mentioned_movies) > 0:
            # Đề xuất dựa trên phim đã đề cập gần đây nhất
            movie_title = mentioned_movies[-1]
            recommendations = chatbot.recommender.get_recommendations(movie_title=movie_title, top_n=count)
        # elif liked_movies and len(liked_movies) > 0:
        #     # Đề xuất dựa trên phim đã thích gần đây nhất
        #     movie_id = liked_movies[-1]
        #     # Giả sử bạn có phương thức để lấy tên phim từ ID
        #     movie = chatbot.recommender.get_movie_by_id(movie_id)
        #     if movie and 'title' in movie:
        #         recommendations = chatbot.recommender.get_recommendations(movie_title=movie['title'], top_n=count)
        #     else:
        #         # Nếu không tìm thấy phim theo ID, chuyển sang tiêu chí tiếp theo
        #         if genres:
        #             recommendations = chatbot.recommender.get_genre_recommendations(genres, film_type, count)
        #         elif film_type:
        #             recommendations = chatbot.recommender.get_film_type_recommendations(film_type, count)
        #         else:
        #             recommendations = chatbot.recommender.get_popular_movies(count)
        elif genres:
            recommendations = chatbot.recommender.get_genre_recommendations(genres, film_type, count)
        elif film_type:
            recommendations = chatbot.recommender.get_film_type_recommendations(film_type, count)
        else:
            recommendations = chatbot.recommender.get_popular_movies(count)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'params': {
                'liked_movies': liked_movies,
                'disliked_movies': disliked_movies,
                'genres': genres,
                'film_types': film_types_list,
                'mentioned_movies': mentioned_movies,
                'count': count
            }
        })
    except Exception as e:
        logger.error(f"Lỗi khi lấy đề xuất: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

    except Exception as e:
        logger.error(f"Lỗi khi lấy đề xuất: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/search', methods=['GET'])
def search_movies():
    try:
        global chatbot
        # Kiểm tra xem chatbot đã được khởi tạo chưa
        if chatbot is None or chatbot.recommender is None:
            return jsonify({
                'success': False,
                'error': 'Hệ thống đề xuất chưa được khởi tạo, vui lòng thử lại sau.'
            }), 500
        
        query = request.args.get('query', '')
        if not query:
            return jsonify({'error': 'Không có truy vấn tìm kiếm được cung cấp'}), 400
        
        count = int(request.args.get('count', 5))
        results = chatbot.recommender.search_movies(query, count)
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/genres', methods=['GET'])
def get_genres():
    """Trả về danh sách thể loại phim."""
    return jsonify({
        'success': True,
        'genres': VIETNAMESE_GENRES
    })

@app.route('/film-types', methods=['GET'])
def get_film_types():
    """Trả về danh sách loại phim."""
    return jsonify({
        'success': True,
        'film_types': FILM_TYPES
    })
# Thêm route để đặt lại sở thích
@app.route('/reset-preferences', methods=['POST', 'OPTIONS'])
def reset_preferences():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        reset_genres = data.get('reset_genres', True)
        reset_film_types = data.get('reset_film_types', True)
        reset_interactions = data.get('reset_interactions', True)
        reset_history = data.get('reset_history', True)
        
        session_id = session.get('session_id')
        user_id = session.get('user_id')
        
        success = preference_manager.reset_preferences(
            user_id, session_id, 
            reset_genres, reset_film_types, 
            reset_interactions, reset_history
        )
        
        # Cập nhật chatbot
        if success:
            chatbot.user_preferences = {
                "genres": [],
                "film_types": [],
                "liked_movies": [],
                "disliked_movies": [],
                "mentioned_movies": []
            }
        
        return jsonify({
            'success': success,
            'message': 'Đã đặt lại sở thích người dùng thành công' if success else 'Lỗi khi đặt lại sở thích'
        })
    except Exception as e:
        logger.error(f"Lỗi khi đặt lại sở thích: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
# Endpoint lấy sở thích người dùng
@app.route('/user-preferences', methods=['GET', 'OPTIONS'])
def get_user_preferences():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Lấy session_id từ cookie
        session_id = session.get('session_id')
        user_id = session.get('user_id')  # None nếu chưa đăng nhập
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy session_id'
            }), 400
        
        # Lấy sở thích từ cơ sở dữ liệu
        if preference_manager:
            preferences = preference_manager.get_user_preferences(user_id, session_id)
            
            # Chuyển đổi sang định dạng đơn giản hơn cho frontend
            simplified_preferences = {
                "genres": [p["genre"] for p in preferences.get("genres", [])],
                "film_types": [p["film_type"] for p in preferences.get("film_types", [])],
                "liked_movies": preferences.get("liked_movies", []),
                "disliked_movies": preferences.get("disliked_movies", []),
                "viewed_movies": preferences.get("viewed_movies", [])
            }
            
            return jsonify({
                'success': True,
                'preferences': simplified_preferences
            })
        else:
            # Fallback nếu không có preference_manager
            return jsonify({
                'success': True,
                'preferences': {
                    "genres": [],
                    "film_types": [],
                    "liked_movies": [],
                    "disliked_movies": [],
                    "viewed_movies": []
                }
            })
    except Exception as e:
        logger.error(f"Lỗi khi lấy sở thích người dùng: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
# Endpoint ghi nhận tương tác với phim
@app.route('/movie-interaction', methods=['POST', 'OPTIONS'])
def movie_interaction():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        movie_id = data.get('movie_id')
        interaction_type = data.get('interaction_type')
        movie_title = data.get('movie_title')
        
        if not movie_id or not interaction_type:
            return jsonify({
                'success': False,
                'error': 'Thiếu thông tin bắt buộc'
            }), 400
        
        # Lấy session_id từ cookie
        session_id = session.get('session_id')
        user_id = session.get('user_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy session_id'
            }), 400
        
        # Ghi nhận tương tác
        if preference_manager:
            success = preference_manager.record_movie_interaction(
                movie_id, interaction_type, user_id, session_id
            )
            
            # Cập nhật sở thích trong chatbot
            if success and chatbot:
                if interaction_type == 'like' and movie_id not in chatbot.user_preferences["liked_movies"]:
                    chatbot.user_preferences["liked_movies"].append(movie_id)
                    
                    # Nếu có movie_title, thử cập nhật thể loại
                    if movie_title:
                        idx = chatbot.recommender.get_movie_index(movie_title)
                        if idx is not None:
                            try:
                                movie_row = chatbot.recommender.df.iloc[idx]
                                if 'genre' in movie_row and movie_row['genre']:
                                    genres = str(movie_row['genre']).split('|')
                                    for genre in genres:
                                        genre = genre.strip()
                                        if genre and genre not in chatbot.user_preferences["genres"]:
                                            chatbot.user_preferences["genres"].append(genre)
                                            if preference_manager:
                                                preference_manager.add_genre_preference(genre, user_id, session_id)
                            except Exception as e:
                                logger.error(f"Lỗi khi cập nhật thể loại từ phim: {str(e)}")
                
                elif interaction_type == 'dislike':
                    if movie_id in chatbot.user_preferences["liked_movies"]:
                        chatbot.user_preferences["liked_movies"].remove(movie_id)
                    if movie_id not in chatbot.user_preferences["disliked_movies"]:
                        chatbot.user_preferences["disliked_movies"].append(movie_id)
            
            # Lấy sở thích cập nhật
            preferences = preference_manager.get_user_preferences(user_id, session_id)
            simplified_preferences = {
                "genres": [p["genre"] for p in preferences.get("genres", [])],
                "film_types": [p["film_type"] for p in preferences.get("film_types", [])],
                "liked_movies": preferences.get("liked_movies", []),
                "disliked_movies": preferences.get("disliked_movies", []),
                "viewed_movies": preferences.get("viewed_movies", [])
            }
            
            return jsonify({
                'success': success,
                'message': f'Đã ghi nhận tương tác {interaction_type} với phim {movie_id}',
                'preferences': simplified_preferences
            })
        else:
            # Fallback nếu không có preference_manager
            return jsonify({
                'success': False,
                'error': 'Hệ thống sở thích không khả dụng'
            }), 500
    except Exception as e:
        logger.error(f"Lỗi khi ghi nhận tương tác: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/remove-preference', methods=['POST', 'OPTIONS'])
def remove_preference():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        preference_type = data.get('preference_type')
        preference_value = data.get('preference_value')
        
        if not preference_type or preference_value is None:
            return jsonify({
                'success': False,
                'error': 'Thiếu thông tin bắt buộc'
            }), 400
        
        # Lấy session_id từ cookie
        session_id = session.get('session_id')
        user_id = session.get('user_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Không tìm thấy session_id'
            }), 400
        
        # Xóa sở thích
        if preference_manager:
            success = False
            
            if preference_type == 'genre':
                # Xóa thể loại
                success = preference_manager.remove_genre_preference(preference_value, user_id, session_id)
                
                # Cập nhật chatbot
                if success and chatbot and preference_value in chatbot.user_preferences["genres"]:
                    chatbot.user_preferences["genres"].remove(preference_value)
            
            elif preference_type == 'movie':
                # Xóa tương tác với phim
                success = preference_manager.remove_movie_interaction(preference_value, "like", user_id, session_id)
                
                # Cập nhật chatbot
                if success and chatbot and preference_value in chatbot.user_preferences["liked_movies"]:
                    chatbot.user_preferences["liked_movies"].remove(preference_value)
            
            # Lấy sở thích cập nhật
            preferences = preference_manager.get_user_preferences(user_id, session_id)
            simplified_preferences = {
                "genres": [p["genre"] for p in preferences.get("genres", [])],
                "film_types": [p["film_type"] for p in preferences.get("film_types", [])],
                "liked_movies": preferences.get("liked_movies", []),
                "disliked_movies": preferences.get("disliked_movies", [])
            }
            
            return jsonify({
                'success': success,
                'message': f'Đã xóa sở thích {preference_type}: {preference_value}',
                'preferences': simplified_preferences
            })
        else:
            # Fallback nếu không có preference_manager
            return jsonify({
                'success': False,
                'error': 'Hệ thống sở thích không khả dụng'
            }), 500
    except Exception as e:
        logger.error(f"Lỗi khi xóa sở thích: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Endpoint ghi nhận tương tác với phim
# @app.route('/movie-interaction', methods=['POST', 'OPTIONS'])
# def movie_interaction():
#     if request.method == 'OPTIONS':
#         return '', 204
    
#     try:
#         data = request.get_json()
#         movie_id = data.get('movie_id')
#         interaction_type = data.get('interaction_type')
#         movie_title = data.get('movie_title')
        
#         if not movie_id or not interaction_type:
#             return jsonify({
#                 'success': False,
#                 'error': 'Thiếu thông tin bắt buộc'
#             }), 400
        
#         # Lấy session_id từ cookie
#         session_id = session.get('session_id')
#         user_id = session.get('user_id')
        
#         if not session_id:
#             return jsonify({
#                 'success': False,
#                 'error': 'Không tìm thấy session_id'
#             }), 400
        
#         # Ghi nhận tương tác
#         if preference_manager:
#             success = preference_manager.record_movie_interaction(
#                 movie_id, interaction_type, user_id, session_id
#             )
            
#             # Cập nhật sở thích trong chatbot
#             if success and chatbot:
#                 if interaction_type == 'like' and movie_id not in chatbot.user_preferences["liked_movies"]:
#                     chatbot.user_preferences["liked_movies"].append(movie_id)
                    
#                     # Nếu có movie_title, thử cập nhật thể loại
#                     if movie_title:
#                         idx = chatbot.recommender.get_movie_index(movie_title)
#                         if idx is not None:
#                             try:
#                                 movie_row = chatbot.recommender.df.iloc[idx]
#                                 if 'genre' in movie_row and movie_row['genre']:
#                                     genres = str(movie_row['genre']).split('|')
#                                     for genre in genres:
#                                         genre = genre.strip()
#                                         if genre and genre not in chatbot.user_preferences["genres"]:
#                                             chatbot.user_preferences["genres"].append(genre)
#                                             if preference_manager:
#                                                 preference_manager.add_genre_preference(genre, user_id, session_id)
#                             except Exception as e:
#                                 logger.error(f"Lỗi khi cập nhật thể loại từ phim: {str(e)}")
                
#                 elif interaction_type == 'dislike':
#                     if movie_id in chatbot.user_preferences["liked_movies"]:
#                         chatbot.user_preferences["liked_movies"].remove(movie_id)
#                     if movie_id not in chatbot.user_preferences["disliked_movies"]:
#                         chatbot.user_preferences["disliked_movies"].append(movie_id)
            
#             # Lấy sở thích cập nhật
#             preferences = preference_manager.get_user_preferences(user_id, session_id)
#             simplified_preferences = {
#                 "genres": [p["genre"] for p in preferences.get("genres", [])],
#                 "film_types": [p["film_type"] for p in preferences.get("film_types", [])],
#                 "liked_movies": preferences.get("liked_movies", []),
#                 "disliked_movies": preferences.get("disliked_movies", []),
#                 "viewed_movies": preferences.get("viewed_movies", [])
#             }
            
#             return jsonify({
#                 'success': success,
#                 'message': f'Đã ghi nhận tương tác {interaction_type} với phim {movie_id}',
#                 'preferences': simplified_preferences
#             })
#         else:
#             # Fallback nếu không có preference_manager
#             return jsonify({
#                 'success': False,
#                 'error': 'Hệ thống sở thích không khả dụng'
#             }), 500
#     except Exception as e:
#         logger.error(f"Lỗi khi ghi nhận tương tác: {str(e)}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

# Chạy ứng dụng Flask
if __name__ == "__main__":
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
    logger.info(f"Khởi động ứng dụng Flask trên cổng {port}, debug={debug}")
    
    app.run(debug=debug, port=port, host='0.0.0.0', threaded=True)
