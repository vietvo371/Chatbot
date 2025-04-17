import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from typing import List, Dict, Any, Optional
import logging

# Thiết lập logging để dễ dàng debug
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

class MovieRecommender:
    def __init__(self, csv_path):
        """Khởi tạo hệ thống đề xuất phim từ file CSV."""
        self.df = self.load_and_preprocess_data(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.features = None
        self.similarity_matrix = None
        self.create_features()
    
    def load_and_preprocess_data(self, csv_path):
        """Đọc file CSV và tiền xử lý dữ liệu."""
        try:
            logger.info(f"Đang đọc dữ liệu từ {csv_path}")
            if not os.path.exists(csv_path):
                logger.error(f"File {csv_path} không tồn tại!")
                # Tạo file mẫu nếu không tồn tại
                self.create_sample_csv(csv_path)
                logger.info(f"Đã tạo file mẫu {csv_path}")
                
            df = pd.read_csv(csv_path)
            logger.info(f"Đã đọc file CSV thành công, có {len(df)} dòng dữ liệu")
            
            required_columns = self.check_required_columns(df)
            
            # Điền các giá trị NaN
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].fillna('')
                else:
                    df[col] = df[col].fillna(0)
            
            # Tạo cột đặc trưng kết hợp cho việc lọc dựa trên nội dung
            df['combined_features'] = self.create_combined_features(df)
            
            # In thông tin về dữ liệu
            logger.info(f"Đã tải {len(df)} bộ phim")
            logger.info(f"Các cột: {df.columns.tolist()}")
            
            return df
        except Exception as e:
            logger.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
            raise
    
    def create_sample_csv(self, csv_path):
        """Tạo file CSV mẫu nếu không tồn tại."""
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
        movies_df.to_csv(csv_path, index=False)
    
    def check_required_columns(self, df):
        """Kiểm tra các cột cần thiết trong dataframe."""
        required = ['title']  # Cột bắt buộc: tên phim
        
        # Các cột khuyến nghị
        recommended = [
            'genre',        # Thể loại 
            'film_type',    # Loại phim (Phim Lẻ, Phim Bộ, Phim Hoạt Hình)
            'director',     # Đạo diễn
            'actors',       # Diễn viên
            'plot',         # Nội dung
            'year',         # Năm phát hành
            'rating',       # Đánh giá
            'country'       # Quốc gia
        ]
        
        # Kiểm tra các cột bắt buộc
        missing_required = [col for col in required if col not in df.columns]
        if missing_required:
            raise ValueError(f"Thiếu các cột bắt buộc: {missing_required}")
        
        # Kiểm tra các cột khuyến nghị và thêm cột rỗng nếu thiếu
        for col in recommended:
            if col not in df.columns:
                logger.warning(f"Không tìm thấy cột '{col}'. Thêm cột trống.")
                df[col] = ''
        
        return required + [col for col in recommended if col in df.columns]
    
    def create_combined_features(self, df):
        """Tạo chuỗi đặc trưng kết hợp cho mỗi bộ phim."""
        features = []
        
        for _, row in df.iterrows():
            combined = ""
            
            # Thêm tiêu đề (với trọng số cao hơn bằng cách lặp lại)
            if 'title' in df.columns and row['title']:
                combined += str(row['title']) + " " + str(row['title']) + " "
            
            # Thêm thể loại (với trọng số cao hơn)
            if 'genre' in df.columns and row['genre']:
                genres = str(row['genre']).replace(',', ' ').replace('|', ' ')
                combined += genres + " " + genres + " "
            
            # Thêm loại phim
            if 'film_type' in df.columns and row['film_type']:
                combined += str(row['film_type']) + " "
            
            # Thêm đạo diễn
            if 'director' in df.columns and row['director']:
                combined += str(row['director']) + " "
            
            # Thêm diễn viên
            if 'actors' in df.columns and row['actors']:
                combined += str(row['actors']).replace(',', ' ') + " "
            
            # Thêm cốt truyện
            if 'plot' in df.columns and row['plot']:
                combined += str(row['plot']) + " "
            
            # Thêm năm
            if 'year' in df.columns and row['year']:
                combined += str(row['year']) + " "
            
            # Thêm quốc gia
            if 'country' in df.columns and row['country']:
                combined += str(row['country']) + " "
            
            features.append(combined.strip())
        
        return features
    
    def create_features(self):
        """Tạo đặc trưng TF-IDF và ma trận tương đồng."""
        try:
            # Tạo đặc trưng TF-IDF
            logger.info("Đang tạo đặc trưng TF-IDF...")
            self.features = self.vectorizer.fit_transform(self.df['combined_features'])
            
            # Tính toán ma trận tương đồng
            logger.info("Đang tính toán ma trận tương đồng...")
            self.similarity_matrix = cosine_similarity(self.features)
            
            logger.info(f"Kích thước ma trận đặc trưng: {self.features.shape}")
            logger.info(f"Kích thước ma trận tương đồng: {self.similarity_matrix.shape}")
        except Exception as e:
            logger.error(f"Lỗi khi tạo đặc trưng: {str(e)}")
            raise
    
    def get_movie_index(self, movie_title):
        """Lấy chỉ số của phim theo tiêu đề."""
        try:
            # Tìm kiếm chính xác
            matches = self.df[self.df['title'].str.lower() == movie_title.lower()]
            if not matches.empty:
                return matches.index[0]
            
            # Nếu không tìm thấy kết quả chính xác, thử tìm kiếm một phần
            matches = self.df[self.df['title'].str.lower().str.contains(movie_title.lower(), na=False)]
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
        """Lấy các phim tương tự dựa trên tiêu đề phim."""
        idx = self.get_movie_index(movie_title)
        if idx is None:
            logger.warning(f"Không tìm thấy phim: {movie_title}")
            return []
        
        try:
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
                recommendations.append(movie_data)
            
            return recommendations
        except Exception as e:
            logger.error(f"Lỗi khi lấy phim tương tự: {str(e)}")
            return []
    
    def get_genre_recommendations(self, genres, film_type=None, top_n=5):
        """Lấy đề xuất dựa trên thể loại và loại phim (nếu có)."""
        try:
            # Đảm bảo genres là list
            if isinstance(genres, str):
                genres = [genres]
            
            logger.info(f"Tìm phim theo thể loại: {genres}, loại phim: {film_type}")
            
            # Tính điểm trùng khớp thể loại cho mỗi bộ phim
            genre_scores = []
            for idx, row in self.df.iterrows():
                if 'genre' not in self.df.columns or not row['genre']:
                    continue
                
                # Kiểm tra loại phim nếu được chỉ định
                if film_type and 'film_type' in self.df.columns:
                    if row['film_type'] != film_type:
                        continue
                
                movie_genres = str(row['genre']).split('|')
                movie_genres = [g.strip() for g in movie_genres]
                
                # Tính điểm trùng khớp dựa trên số lượng thể loại trùng khớp
                match_score = len(set(genres).intersection(set(movie_genres)))
                if match_score > 0:
                    genre_scores.append((idx, match_score))
            
            # Sắp xếp theo điểm trùng khớp
            genre_scores = sorted(genre_scores, key=lambda x: x[1], reverse=True)
            
            logger.info(f"Số lượng phim phù hợp: {len(genre_scores)}")
            
            # Lấy top N kết quả trùng khớp
            top_matches = genre_scores[:top_n]
            
            # Trả về thông tin chi tiết của phim
            recommendations = []
            for idx, _ in top_matches:
                movie_data = self.df.iloc[idx].to_dict()
                # Loại bỏ combined_features khỏi đầu ra
                if 'combined_features' in movie_data:
                    del movie_data['combined_features']
                recommendations.append(movie_data)
            
            return recommendations
        except Exception as e:
            logger.error(f"Lỗi khi lấy đề xuất theo thể loại: {str(e)}")
            return []
    
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
            
            # Sắp xếp theo đánh giá nếu có
            if 'rating' in filtered_df.columns:
                filtered_df = filtered_df.sort_values(by='rating', ascending=False)
            
            # Lấy top N phim
            top_movies = filtered_df.head(top_n)
            
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
            logger.error(f"Lỗi khi lấy đề xuất theo loại phim: {str(e)}")
            return []
    
    def get_popular_movies(self, top_n=5):
        """Lấy các phim phổ biến dựa trên đánh giá."""
        try:
            if 'rating' in self.df.columns:
                top_movies = self.df.sort_values(by='rating', ascending=False).head(top_n)
            else:
                # Nếu không có cột đánh giá, trả về các phim ngẫu nhiên
                top_movies = self.df.sample(n=min(top_n, len(self.df)))
            
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
            return []
    
    def search_movies(self, query, top_n=5):
        """Tìm kiếm phim theo tiêu đề, diễn viên, đạo diễn hoặc thể loại."""
        try:
            query = query.lower()
            
            # Tìm kiếm trong các cột khác nhau
            title_matches = self.df[self.df['title'].str.lower().str.contains(query, regex=False, na=False)]
            
            actor_matches = pd.DataFrame()
            if 'actors' in self.df.columns:
                actor_matches = self.df[self.df['actors'].str.lower().str.contains(query, regex=False, na=False)]
            
            director_matches = pd.DataFrame()
            if 'director' in self.df.columns:
                director_matches = self.df[self.df['director'].str.lower().str.contains(query, regex=False, na=False)]
            
            genre_matches = pd.DataFrame()
            if 'genre' in self.df.columns:
                genre_matches = self.df[self.df['genre'].str.lower().str.contains(query, regex=False, na=False)]
            
            country_matches = pd.DataFrame()
            if 'country' in self.df.columns:
                country_matches = self.df[self.df['country'].str.lower().str.contains(query, regex=False, na=False)]
            
            # Kết hợp tất cả các kết quả trùng khớp và loại bỏ trùng lặp
            all_matches = pd.concat([title_matches, actor_matches, director_matches, genre_matches, country_matches]).drop_duplicates()
            
            # Sắp xếp theo đánh giá nếu có
            if 'rating' in all_matches.columns and not all_matches.empty:
                all_matches = all_matches.sort_values(by='rating', ascending=False)
            
            # Lấy top N kết quả trùng khớp
            top_matches = all_matches.head(top_n)
            
            recommendations = []
            for _, row in top_matches.iterrows():
                movie_data = row.to_dict()
                # Loại bỏ combined_features khỏi đầu ra
                if 'combined_features' in movie_data:
                    del movie_data['combined_features']
                recommendations.append(movie_data)
            
            return recommendations
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm phim: {str(e)}")
            return []


class MovieRecommendationChatbot:
    def __init__(self, csv_path):
        """Khởi tạo chatbot với hệ thống đề xuất phim."""
        logger.info(f"Khởi tạo chatbot với dữ liệu từ {csv_path}")
        self.recommender = MovieRecommender(csv_path)
        self.history = []
        self.user_preferences = {
            "genres": [], 
            "film_types": [],
            "liked_movies": [], 
            "disliked_movies": [],
            "mentioned_movies": []
        }
    
    def process_message(self, message):
        """Xử lý tin nhắn của người dùng và tạo phản hồi."""
        try:
            logger.info(f"Xử lý tin nhắn: {message}")
            # Thêm tin nhắn người dùng vào lịch sử
            self.history.append({"role": "user", "content": message})
            
            # Trích xuất ý định và thực thể từ tin nhắn
            intent = self.extract_intent(message)
            entities = self.extract_entities(message)
            
            logger.info(f"Ý định: {intent}")
            logger.info(f"Thực thể: {entities}")
            
            # Tạo phản hồi dựa trên ý định
            response = self.generate_response(intent, entities, message)
            
            # Thêm phản hồi của bot vào lịch sử
            self.history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            error_msg = f"Đã xảy ra lỗi khi xử lý tin nhắn: {str(e)}"
            logger.error(error_msg)
            self.history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def extract_intent(self, message):
        """Trích xuất ý định của người dùng từ tin nhắn."""
        message = message.lower()
        
        # Kiểm tra xem tin nhắn chỉ chứa tên thể loại không
        for genre in VIETNAMESE_GENRES:
            genre_lower = genre.lower()
            if message == genre_lower or message == f"phim {genre_lower}":
                if genre not in self.user_preferences["genres"]:
                    self.user_preferences["genres"].append(genre)
                return "recommend"
        
        # Kiểm tra loại phim
        for film_type in FILM_TYPES:
            if message == film_type.lower():
                if film_type not in self.user_preferences["film_types"]:
                    self.user_preferences["film_types"].append(film_type)
                return "recommend"
        
        # Kiểm tra các ý định khác
        if re.search(r'đề xuất|gợi ý|giới thiệu|tư vấn|recommend|suggest|show|cho.*xem|xem.*phim', message):
            return "recommend"
        elif re.search(r'thích|yêu thích|like|love|enjoy|favorite', message):
            return "like"
        elif re.search(r'không thích|ghét|dislike|hate|don\'t like', message):
            return "dislike"
        elif re.search(r'tìm|kiếm|search|find|looking for', message):
            return "search"
        elif re.search(r'giúp|trợ giúp|help|how can you|what can you do', message):
            return "help"
        else:
            return "chat"
    
    def extract_entities(self, message):
        """Trích xuất thực thể như tiêu đề phim, thể loại và loại phim từ tin nhắn."""
        entities = {
            "movie_title": None,
            "genres": [],
            "film_type": None
        }
        
        message_lower = message.lower()
        
        # Trích xuất tiêu đề phim - tìm kiếm dấu ngoặc kép hoặc từ khóa phim/film
        title_match = re.search(r'"([^"]+)"|\'([^\']+)\'|phim "([^"]+)"|film "([^"]+)"', message)
        if title_match:
            # Lấy nhóm đầu tiên không rỗng
            for group in title_match.groups():
                if group:
                    entities["movie_title"] = group
                    break
        
        # Trích xuất thể loại
        for genre in VIETNAMESE_GENRES:
            if re.search(r'\b' + genre.lower() + r'\b', message_lower):
                entities["genres"].append(genre)
        
        # Trích xuất loại phim
        for film_type in FILM_TYPES:
            if re.search(r'\b' + film_type.lower() + r'\b', message_lower):
                entities["film_type"] = film_type
                break
        
        return entities
    
    def generate_response(self, intent, entities, original_message):
        """Tạo phản hồi dựa trên ý định và thực thể."""
        if intent == "recommend":
            return self._handle_recommendation(entities)
        elif intent == "like":
            return self._handle_like(entities)
        elif intent == "dislike":
            return self._handle_dislike(entities)
        elif intent == "search":
            return self._handle_search(original_message)
        elif intent == "help":
            return self._handle_help()
        else:
            return self._handle_chat(original_message)
    
    def _handle_recommendation(self, entities):
        """Xử lý yêu cầu đề xuất."""
        # Cập nhật sở thích thể loại nếu có
        if entities["genres"]:
            for genre in entities["genres"]:
                if genre not in self.user_preferences["genres"]:
                    self.user_preferences["genres"].append(genre)
        
        # Cập nhật loại phim nếu có
        if entities["film_type"] and entities["film_type"] not in self.user_preferences["film_types"]:
            self.user_preferences["film_types"].append(entities["film_type"])
        
        if entities["movie_title"]:
            # Cập nhật danh sách phim đã đề cập
            if entities["movie_title"] not in self.user_preferences.get("mentioned_movies", []):
                self.user_preferences["mentioned_movies"] = self.user_preferences.get("mentioned_movies", [])
                self.user_preferences["mentioned_movies"].append(entities["movie_title"])
            
            # Đề xuất dựa trên phim
            recommendations = self.recommender.get_recommendations(movie_title=entities["movie_title"])
            if recommendations:
                response = f"Dựa trên phim \"{entities['movie_title']}\", tôi đề xuất cho bạn các phim sau:\n\n"
                for i, movie in enumerate(recommendations, 1):
                    title = movie.get('title', 'Không có tiêu đề')
                    genre = movie.get('genre', 'Không có thể loại')
                    film_type = movie.get('film_type', '')
                    year = movie.get('year', '')
                    rating = movie.get('rating', '')
                    
                    response += f"{i}. {title}"
                    if year:
                        response += f" ({year})"
                    if film_type:
                        response += f" - {film_type}"
                    if genre:
                        response += f" - {genre}"
                    if rating:
                        response += f" - Đánh giá: {rating}/10"
                    response += "\n"
                
                response += "\nBạn có thích một trong những đề xuất này không?"
                return response
            else:
                return f"Xin lỗi, tôi không tìm thấy phim \"{entities['movie_title']}\" trong cơ sở dữ liệu của mình. Bạn có thể thử một bộ phim khác không?"
        
        elif entities["genres"]:
            # Đề xuất dựa trên thể loại và loại phim (nếu có)
            recommendations = self.recommender.get_genre_recommendations(
                entities["genres"], 
                entities["film_type"]
            )
            
            genres_text = ", ".join(entities["genres"])
            film_type_text = f" thuộc loại {entities['film_type']}" if entities["film_type"] else ""
            
            if recommendations:
                response = f"Dựa trên thể loại {genres_text}{film_type_text}, tôi đề xuất cho bạn các phim sau:\n\n"
                for i, movie in enumerate(recommendations, 1):
                    title = movie.get('title', 'Không có tiêu đề')
                    genre = movie.get('genre', 'Không có thể loại')
                    film_type = movie.get('film_type', '')
                    year = movie.get('year', '')
                    rating = movie.get('rating', '')
                    
                    response += f"{i}. {title}"
                    if year:
                        response += f" ({year})"
                    if film_type:
                        response += f" - {film_type}"
                    if genre:
                        response += f" - {genre}"
                    if rating:
                        response += f" - Đánh giá: {rating}/10"
                    response += "\n"
                
                response += "\nBạn có thích một trong những đề xuất này không?"
                return response
            else:
                return f"Xin lỗi, tôi không tìm thấy phim nào thuộc thể loại {genres_text}{film_type_text} trong cơ sở dữ liệu của mình. Bạn có thể thử thể loại khác không?"
        
        elif entities["film_type"]:
            # Đề xuất dựa trên loại phim
            recommendations = self.recommender.get_film_type_recommendations(entities["film_type"])
            
            if recommendations:
                response = f"Dựa trên loại {entities['film_type']}, tôi đề xuất cho bạn các phim sau:\n\n"
                for i, movie in enumerate(recommendations, 1):
                    title = movie.get('title', 'Không có tiêu đề')
                    genre = movie.get('genre', 'Không có thể loại')
                    year = movie.get('year', '')
                    rating = movie.get('rating', '')
                    
                    response += f"{i}. {title}"
                    if year:
                        response += f" ({year})"
                    if genre:
                        response += f" - {genre}"
                    if rating:
                        response += f" - Đánh giá: {rating}/10"
                    response += "\n"
                
                response += "\nBạn có thích một trong những đề xuất này không?"
                return response
            else:
                return f"Xin lỗi, tôi không tìm thấy phim nào thuộc loại {entities['film_type']} trong cơ sở dữ liệu của mình. Bạn có thể thử loại phim khác không?"
        
        else:
            # Nếu không có tiêu chí cụ thể, đề xuất phim phổ biến hoặc dựa trên sở thích người dùng
            if self.user_preferences["genres"]:
                film_type = None
                if self.user_preferences["film_types"]:
                    film_type = self.user_preferences["film_types"][-1]
                    
                recommendations = self.recommender.get_genre_recommendations(
                    self.user_preferences["genres"], 
                    film_type
                )
                
                genres_text = ", ".join(self.user_preferences["genres"])
                film_type_text = f" thuộc loại {film_type}" if film_type else ""
                source = f"thể loại yêu thích của bạn ({genres_text}{film_type_text})"
                
            elif self.user_preferences["film_types"]:
                film_type = self.user_preferences["film_types"][-1]
                recommendations = self.recommender.get_film_type_recommendations(film_type)
                source = f"loại phim yêu thích của bạn ({film_type})"
                
            elif self.user_preferences["liked_movies"] and len(self.user_preferences["liked_movies"]) > 0:
                # Sử dụng phim được yêu thích gần đây nhất
                recommendations = self.recommender.get_recommendations(
                    movie_title=self.user_preferences["liked_movies"][-1]
                )
                source = f"phim bạn đã thích (\"{self.user_preferences['liked_movies'][-1]}\")"
            
            else:
                recommendations = self.recommender.get_popular_movies()
                source = "các phim phổ biến"
            
            if recommendations:
                response = f"Dựa trên {source}, tôi đề xuất cho bạn các phim sau:\n\n"
                for i, movie in enumerate(recommendations, 1):
                    title = movie.get('title', 'Không có tiêu đề')
                    genre = movie.get('genre', 'Không có thể loại')
                    film_type = movie.get('film_type', '')
                    year = movie.get('year', '')
                    rating = movie.get('rating', '')
                    
                    response += f"{i}. {title}"
                    if year:
                        response += f" ({year})"
                    if film_type:
                        response += f" - {film_type}"
                    if genre:
                        response += f" - {genre}"
                    if rating:
                        response += f" - Đánh giá: {rating}/10"
                    response += "\n"
                
                response += "\nBạn có thích một trong những đề xuất này không?"
                return response
            else:
                return "Xin lỗi, tôi không thể đưa ra đề xuất nào vào lúc này. Bạn có thể cho tôi biết bạn thích thể loại phim nào hoặc một bộ phim bạn đã thích không?"
    
    def _handle_like(self, entities):
        """Xử lý khi người dùng thích một bộ phim."""
        if entities["movie_title"]:
            movie_title = entities["movie_title"]
            
            # Thêm vào danh sách phim đã thích nếu chưa có
            if movie_title not in self.user_preferences["liked_movies"]:
                self.user_preferences["liked_movies"].append(movie_title)
            
            # Trích xuất thể loại từ phim nếu có thể
            idx = self.recommender.get_movie_index(movie_title)
            if idx is not None and 'genre' in self.recommender.df.columns:
                movie_genres = str(self.recommender.df.iloc[idx]['genre']).split('|')
                movie_genres = [g.strip() for g in movie_genres]
                
                # Cập nhật sở thích thể loại của người dùng
                for genre in movie_genres:
                    if genre and genre not in self.user_preferences["genres"]:
                        self.user_preferences["genres"].append(genre)
            
            # Lấy đề xuất dựa trên phim này
            recommendations = self.recommender.get_recommendations(movie_title=movie_title)
            
            if recommendations:
                response = f"Tôi rất vui khi bạn thích \"{movie_title}\"! Dựa trên bộ phim này, bạn có thể thích:\n\n"
                for i, movie in enumerate(recommendations[:3], 1):
                    title = movie.get('title', 'Không có tiêu đề')
                    genre = movie.get('genre', 'Không có thể loại')
                    film_type = movie.get('film_type', '')
                    year = movie.get('year', '')
                    
                    response += f"{i}. {title}"
                    if year:
                        response += f" ({year})"
                    if film_type:
                        response += f" - {film_type}"
                    if genre:
                        response += f" - {genre}"
                    response += "\n"
                
                return response
            else:
                return f"Tôi đã ghi nhận rằng bạn thích \"{movie_title}\". Tôi sẽ sử dụng thông tin này để đưa ra đề xuất tốt hơn trong tương lai."
        else:
            return "Tôi rất vui khi bạn thích nó! Bạn có thể cho tôi biết tên phim cụ thể mà bạn đã thích không?"
    
    def _handle_dislike(self, entities):
        """Xử lý khi người dùng không thích một bộ phim."""
        if entities["movie_title"]:
            movie_title = entities["movie_title"]
            
            # Thêm vào danh sách phim không thích
            if movie_title not in self.user_preferences["disliked_movies"]:
                self.user_preferences["disliked_movies"].append(movie_title)
            
            # Xóa khỏi danh sách phim đã thích nếu có
            if movie_title in self.user_preferences["liked_movies"]:
                self.user_preferences["liked_movies"].remove(movie_title)
            
            return f"Tôi đã ghi nhận rằng bạn không thích \"{movie_title}\". Tôi sẽ tránh đề xuất phim tương tự trong tương lai. Bạn có thể cho tôi biết loại phim bạn thích không?"
        else:
            return "Tôi xin lỗi nếu đề xuất của tôi không phù hợp. Bạn có thể cho tôi biết tên phim cụ thể mà bạn không thích không? Hoặc cho tôi biết bạn thích loại phim nào?"
    
    def _handle_search(self, message):
        """Xử lý yêu cầu tìm kiếm."""
        # Trích xuất truy vấn tìm kiếm
        search_query = re.sub(r'tìm|kiếm|search for|looking for|find|search', '', message, flags=re.IGNORECASE).strip()
        
        if not search_query:
            return "Bạn muốn tìm kiếm phim gì? Hãy cho tôi biết tên phim, diễn viên, đạo diễn hoặc thể loại bạn quan tâm."
        
        # Thực hiện tìm kiếm
        results = self.recommender.search_movies(search_query)
        
        if results:
            response = f"Đây là kết quả tìm kiếm cho \"{search_query}\":\n\n"
            for i, movie in enumerate(results, 1):
                title = movie.get('title', 'Không có tiêu đề')
                genre = movie.get('genre', 'Không có thể loại')
                film_type = movie.get('film_type', '')
                year = movie.get('year', '')
                director = movie.get('director', '')
                actors = movie.get('actors', '')
                
                response += f"{i}. {title}"
                if year:
                    response += f" ({year})"
                if film_type:
                    response += f"\n   Loại: {film_type}"
                if genre:
                    response += f"\n   Thể loại: {genre}"
                if director:
                    response += f"\n   Đạo diễn: {director}"
                if actors:
                    response += f"\n   Diễn viên: {actors}"
                response += "\n\n"
            
            return response
        else:
            return f"Xin lỗi, tôi không tìm thấy kết quả nào cho \"{search_query}\". Bạn có thể thử tìm kiếm khác không?"
    
    def _handle_help(self):
        """Xử lý yêu cầu trợ giúp."""
        help_text = """Tôi là chatbot đề xuất phim. Đây là những gì tôi có thể làm:

1. Đề xuất phim dựa trên một bộ phim bạn thích (ví dụ: "Đề xuất phim giống The Shawshank Redemption")
2. Đề xuất phim theo thể loại (ví dụ: "Tôi muốn xem phim hành động")
3. Đề xuất phim theo loại (ví dụ: "Cho tôi xem phim bộ")
4. Tìm kiếm phim (ví dụ: "Tìm kiếm phim của đạo diễn A")
5. Ghi nhận sở thích của bạn (ví dụ: "Tôi thích phim X")

Các thể loại phim:
- Viễn Tưởng, Tình Cảm, Tài Liệu, Khoa Học, Chiến Tranh
- Âm Nhạc, Chính Kịch, Gia Đình, Thần Thoại, Bí Ẩn
- Tâm Lý, Học Đường, Hành Động, Thể Thao, Kinh Điển
- Hài Hước, Võ Thuật, Phim 18+, Phiêu Lưu, Cổ Trang
- Kinh Dị, Hình Sự

Các loại phim:
- Phim Lẻ, Phim Bộ, Phim Hoạt Hình

Hãy cho tôi biết bạn muốn xem loại phim nào và tôi sẽ giúp bạn tìm ra lựa chọn hoàn hảo!"""
        return help_text
    
    def _handle_chat(self, message):
        """Xử lý trò chuyện chung."""
        # Phản hồi dựa trên quy tắc đơn giản
        message = message.lower()
        
        if re.search(r'(hi|hello|hey|xin chào|chào)', message):
            return "Xin chào! Tôi là chatbot đề xuất phim. Bạn muốn xem loại phim nào hôm nay?"
        
        elif re.search(r'(how are you|khỏe không|thế nào)', message):
            return "Tôi là một chatbot nên không có cảm xúc, nhưng tôi luôn sẵn sàng giúp bạn tìm phim hay! Bạn thích xem thể loại phim nào?"
        
        elif re.search(r'(thank|cảm ơn)', message):
            return "Không có gì! Rất vui khi được giúp đỡ bạn. Bạn có muốn đề xuất thêm phim không?"
        
        elif re.search(r'(bye|goodbye|tạm biệt)', message):
            return "Tạm biệt! Hãy quay lại khi bạn cần đề xuất phim nhé!"
        
        else:
            return "Tôi không chắc mình hiểu ý bạn. Bạn muốn tôi đề xuất phim, tìm kiếm phim, hay bạn muốn chia sẻ về một bộ phim bạn thích? Gõ 'giúp đỡ' để xem tôi có thể làm gì cho bạn."


# Khởi tạo chatbot
csv_path = os.getenv('MOVIE_CSV_PATH', 'vietnamese_movies.csv')
try:
    chatbot = MovieRecommendationChatbot(csv_path)
    logger.info("Đã khởi tạo chatbot thành công!")
except Exception as e:
    error_msg = f"Lỗi khi khởi tạo chatbot: {str(e)}"
    logger.error(error_msg)
    chatbot = None

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'success': True,
        'message': 'Chatbot đề xuất phim đang hoạt động!',
        'endpoints': {
            '/chat': 'POST - Gửi tin nhắn đến chatbot',
            '/recommendations': 'GET - Lấy đề xuất phim',
            '/search': 'GET - Tìm kiếm phim'
        }
    })

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        global chatbot
        # Khởi tạo lại chatbot nếu cần
        if chatbot is None:
            try:
                chatbot = MovieRecommendationChatbot(csv_path)
                logger.info("Khởi tạo lại chatbot thành công!")
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f"Không thể khởi tạo chatbot: {str(e)}"
                }), 500
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Không có tin nhắn được cung cấp'}), 400
        
        user_message = data['message']
        response = chatbot.process_message(user_message)
        
        return jsonify({
            'success': True,
            'response': response,
            'history': chatbot.history,
            'preferences': chatbot.user_preferences
        })
    except Exception as e:
        logger.error(f"Lỗi khi xử lý yêu cầu: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    try:
        global chatbot
        # Khởi tạo lại chatbot nếu cần
        if chatbot is None:
            try:
                chatbot = MovieRecommendationChatbot(csv_path)
                logger.info("Khởi tạo lại chatbot thành công!")
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f"Không thể khởi tạo chatbot: {str(e)}"
                }), 500
        
        movie_title = request.args.get('movie_title')
        genres = request.args.get('genres')
        film_type = request.args.get('film_type')
        
        if genres:
            genres = genres.split(',')
            recommendations = chatbot.recommender.get_genre_recommendations(genres, film_type)
        elif movie_title:
            recommendations = chatbot.recommender.get_recommendations(movie_title=movie_title)
        elif film_type:
            recommendations = chatbot.recommender.get_film_type_recommendations(film_type)
        else:
            recommendations = chatbot.recommender.get_popular_movies()
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
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
        # Khởi tạo lại chatbot nếu cần
        if chatbot is None:
            try:
                chatbot = MovieRecommendationChatbot(csv_path)
                logger.info("Khởi tạo lại chatbot thành công!")
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f"Không thể khởi tạo chatbot: {str(e)}"
                }), 500
        
        query = request.args.get('query', '')
        if not query:
            return jsonify({'error': 'Không có truy vấn tìm kiếm được cung cấp'}), 400
        
        results = chatbot.recommender.search_movies(query)
        
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

if __name__ == "__main__":
    # Đặt đường dẫn đến file CSV phim của bạn
    port = int(os.getenv('PORT', 5001))
    logger.info(f"Khởi động ứng dụng Flask trên cổng {port}...")
    
    # Chạy ứng dụng
    app.run(debug=True, port=port, host='0.0.0.0')
    logger.info("Ứng dụng Flask đã dừng.")
