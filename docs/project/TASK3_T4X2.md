# Task 3 T4 x2, prompt v005

Import `notebooks/run_task3_kaggle_t4x2.ipynb`. Chọn GPU T4 x2 và bật Internet.
Tạo gói mã nguồn từ repository bằng `python scripts/package_kaggle_source.py`; file ZIP được ghi vào `outputs/` và không commit lên Git.
Cập nhật dataset riêng tư bằng `outputs/task3_kaggle_source.zip` mới. Kaggle có thể giải nén ZIP; notebook nhận cả hai dạng. Không giữ hai bộ mã nguồn trong Input.

Giữ dữ liệu 32 mẫu train, public test và requested levels như cũ. Chạy setup, GPU check, data check, model config, smoke rồi hai worker và packaging.

Bản này dùng Qwen3-8B NF4/FP16, không API, không fine-tune. Hai GPU chạy hai shard độc lập. Prompt v005 yêu cầu Level 1-3 diễn đạt bằng lời. Retry chỉ rõ nội dung cần bỏ; sau retry vẫn vi phạm sẽ dùng fallback và ghi dấu, không dừng shard chỉ vì lỗi compliance.

Output mới: `/kaggle/working/task3-qwen3-8b-t4-v5`. Không trộn với kết quả prompt v004. Mỗi shard lưu `predictions.partial.json` sau từng bài; `--resume` chỉ tiếp tục nếu config, dữ liệu và prompt khớp. Nếu thay đổi, dùng thư mục output mới. Lần chạy lại cùng cấu hình bỏ qua bài đã lưu. Cache có thể tái sử dụng các lượt generate đã hoàn tất của bài còn dang dở. Không bảo đảm request đang chạy dở được phục hồi.

Giữ hoặc lưu riêng tư toàn bộ output trước khi kết thúc session Kaggle; session mới cần phục hồi vào đúng đường dẫn. Đừng chạy cell tải revision mới nếu cần tiếp tục revision cũ mà không ghi lại SHA ban đầu.

Sau khi merge, xem `fallback_report.json` và log mỗi shard. Tỷ lệ fallback cao nghĩa là chất lượng chẩn đoán cần đánh giá lại; không coi việc xuất đủ 328 bài là bằng chứng tăng điểm. Checker vẫn là heuristic. Các lỗi GPU, dữ liệu hoặc chẩn đoán JSON không hợp lệ kéo dài vẫn có thể dừng worker; checkpoint bảo toàn các bài đã hoàn tất.

Cảnh báo sampling defaults được xử lý bằng GenerationConfig riêng khi greedy. ETA bỏ thời gian nạp model và số bài được resume, nên phản ánh thời gian suy luận mới tốt hơn. Chưa benchmark trên Kaggle hoặc leaderboard.
