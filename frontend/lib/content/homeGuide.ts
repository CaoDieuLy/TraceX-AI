export const HOME_GUIDE_TITLE = "Hướng dẫn nhập truy vấn";

export const HOME_GUIDE_LINES = [
  "Nhập mô tả văn bản hoặc tải lên hình ảnh mẫu của đối tượng cần tìm.",
  "Bấm Gửi để hệ thống tìm kiếm và trả về Top-N kết quả phù hợp nhất.",
  "Hệ thống hỗ trợ tìm kiếm đối tượng dựa trên đặc điểm ngoại hình như màu sắc, kiểu trang phục, phụ kiện, vật mang theo, và một số hành vi cơ bản của đối tượng.",
  "Ví dụ hỗ trợ: \"Người mặc áo đỏ, quần đen, đội mũ bảo hiểm, xách túi màu đen chạy nhanh\" hoặc \"Người mặc áo trắng, quần jeans, đi bộ\".",
  "Ví dụ chưa hỗ trợ: \"Người mặc áo đỏ đánh nhau với người mặc áo xanh\" hoặc \"Người áo xanh đưa đồ cho người áo trắng\".",
] as const;

export const HOME_GUIDE_FEATURES = [
  {
    title: "Phân tích Video",
    body: "Xử lý hàng nghìn giờ video trong vài phút",
    tone: "blue",
  },
  {
    title: "Nhận dạng đối tượng",
    body: "AI nhận diện khuôn mặt và hành vi chính xác",
    tone: "teal",
  },
  {
    title: "Tốc độ cao",
    body: "Kết quả trả về trong thời gian thực",
    tone: "blue",
  },
] as const;

export const HOME_GUIDE_SUPPORTED_CRITERIA = [
  "Màu sắc trang phục",
  "Kiểu trang phục",
  "Phụ kiện",
  "Vật mang theo",
  "Đi bộ",
  "Chạy",
  "Đứng",
  "Ngồi",
  "Xách túi",
  "Khuôn mặt",
  "Giới tính",
  "Độ tuổi",
] as const;

export const HOME_GUIDE_UNSUPPORTED_EXAMPLES = [
  "Người mặc áo đỏ đánh nhau với người mặc áo xanh",
  "Người áo xanh đưa đồ cho người áo trắng",
] as const;
