from google import genai
import os

# Tải API key từ biến môi trường
api_key = os.getenv('GOOGLE_API_KEY')

# Khởi tạo client
client = genai.Client(api_key=api_key)

# Tạo prompt đơn giản để kiểm tra Gemini xử lý tiếng Việt không dấu
prompt = """
Tiếng Việt không dấu: "hanh dong"
Tiếng Việt có dấu: ?
"""

# Gọi API
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

# In kết quả
print(response.text)
