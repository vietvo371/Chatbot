import json
import uuid
import datetime
from typing import Dict, List, Optional, Union, Any
import mysql.connector
from mysql.connector import pooling

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
