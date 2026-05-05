# User Account Rules

## Default Admin Account
- **Email**: `admin@mcpt.local` (hoặc giá trị trong env `BOOTSTRAP_ADMIN_EMAIL`)
- **Password**: `Admin@123456` (hoặc giá trị trong env `BOOTSTRAP_ADMIN_PASSWORD`)
- **Full Name**: `Administrator`
- **Role**: `SUPER_ADMIN`

## User Roles (rank từ thấp đến cao)
1. `USER` - Người dùng thường
2. `ADMIN` - Quản trị viên
3. `SUPER_ADMIN` - Quản trị viên cao cấp (cao nhất)

## Rules

### 1. Tạo User mới (POST /users)
- Chỉ ADMIN hoặc SUPER_ADMIN mới được tạo user
- Không thể tạo account có role >= role của mình
- Chỉ SUPER_ADMIN mới được tạo account ADMIN

### 2. Sửa User (PATCH /users/{user_id})
- Không thể tự sửa role/active status của chính mình
- Không thể sửa user có role >= role của mình
- Chỉ SUPER_ADMIN mới được sửa ADMIN

### 3. Role Hierarchy
```
SUPER_ADMIN > ADMIN > USER
```

### 4. Environment Variables
```bash
BOOTSTRAP_ADMIN_EMAIL=admin@mcpt.local
BOOTSTRAP_ADMIN_PASSWORD=Admin@123456
BOOTSTRAP_ADMIN_FULL_NAME=Administrator
```
