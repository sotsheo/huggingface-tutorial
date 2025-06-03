# Flask JWT + CRUD News API (MySQL)

## Mô tả
API xác thực người dùng + CRUD tin tức sử dụng Flask, JWT và MySQL.

## Tính năng
- Xác thực JWT: đăng ký, đăng nhập, refresh token
- CRUD bảng News: title, description, image, status
- Cấu trúc Flask rõ ràng, dễ mở rộng

## Cài đặt

### 1. Tạo & kích hoạt môi trường ảo
python -m venv venv  
source venv/bin/activate   # Linux/macOS  
venv\Scripts\activate      # Windows

### 2. Cài đặt thư viện
pip install -r requirements.txt

### 3. Cấu hình file .env
FLASK_ENV=development  
SECRET_KEY=your-secret-key  
JWT_SECRET_KEY=your-jwt-secret  
MYSQL_USER=root  
MYSQL_PASSWORD=yourpassword  
MYSQL_HOST=localhost  
MYSQL_DB=your_database_name

### 4. Tạo CSDL & migrate
flask db init  
flask db migrate -m "Initial migration"  
flask db upgrade

### 5. Chạy server
flask run

## Các API chính

| Endpoint         | Method | Mô tả                  |
|------------------|--------|-------------------------|
| /register        | POST   | Đăng ký người dùng      |
| /login           | POST   | Đăng nhập, trả JWT      |
| /refresh         | POST   | Làm mới access token    |
| /news            | GET    | Lấy danh sách bài viết  |
| /news/<id>       | GET    | Chi tiết bài viết       |
| /news            | POST   | Thêm bài viết (auth)    |
| /news/<id>       | PUT    | Cập nhật bài viết       |
| /news/<id>       | DELETE | Xoá bài viết            |

## Lưu ý
- Các route thêm/sửa/xoá yêu cầu JWT token ở header:
  Authorization: Bearer <access_token>
