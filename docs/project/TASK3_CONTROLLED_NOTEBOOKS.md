# Task 3: ba notebook suy luận độc lập

- `notebooks/Task3_P1_ZeroShot_LevelAware.ipynb`: zero-shot, tự chẩn đoán và sinh phản hồi trong một JSON.
- `notebooks/Task3_P2_RAG.ipynb`: tối đa hai ví dụ phù hợp cùng mức; TF-IDF/cosine trên code, ưu tiên cùng mã đề. Không có ví dụ phù hợp thì dùng prompt zero-shot như P1. Không dùng nhãn của bài đích để truy hồi.
- `notebooks/Task3_P3_PlanThenRender.ipynb`: không RAG; chẩn đoán có cấu trúc, bằng chứng code và nguyên nhân, sau đó sinh phản hồi với nhãn được khóa theo chẩn đoán.

Mỗi notebook nhúng cùng runtime `src/llm_grading/task3_controlled.py` và chạy độc lập, không cần upload source repository. Cấu hình riêng của notebook ở `configs/task3/notebooks/controlled_feedback.yaml`; schema này độc lập với config CLI chung. Sửa runtime/config rồi chạy `python scripts/build_task3_controlled_notebooks.py` để tái tạo theo thứ tự P1, P2, P3. Không sửa riêng các bản runtime nhúng.

## Kết quả kiểm tra dữ liệu

Đã đọc đặc tả assignment, README và `label_space.json`, cấu trúc ZIP public, sidecar mức phản hồi và sample submission Task 3. Prompt mới của người dùng yêu cầu tự chẩn đoán; hướng dẫn cũ trong sample dataset mô tả nhãn cho sẵn không áp dụng cho ba notebook này.

- `public_test.zip` có 328 mẫu, cả `single_problem` và `multi_problem`. Code nằm trong file tham chiếu qua `input.code_file`; không có nhãn đầu vào. Có hai file code rỗng; loader giữ nguyên để mô hình đánh giá, không bỏ mẫu.
- `sample_dataset` có 32 mẫu, 15 EX01 và 17 EX02, với taxonomy ở `input.taxonomy_error` và phản hồi ở `output.feedback`. Phân bố mức 1/2/3 là 4/18/10, không có mức 4.
- Người dùng đã xác nhận `sample_dataset` được phép dùng làm ví dụ minh họa cho P2. Bộ dữ liệu vẫn là tập tham chiếu minh họa, không được gọi là tập train/test chính thức. Cấu hình mặc định ghi nhận quyền sử dụng này bằng `confirm_training_partition=True`; không cần xác nhận lại.
- Nhãn Task 2/Task 3 khớp ở cả 32 mẫu; corpus phủ 9/10 nhãn, thiếu `Lỗi edge case`. Không tự bổ sung nhãn/ví dụ cho phần thiếu.
- Bộ lọc chất lượng nhẹ kiểm tra phản hồi rỗng, nội dung nội bộ, code/cách sửa ở mức bị cấm, bao gồm code inline không có fence hoặc dấu chấm phẩy. Nhờ đó loại cả EX01-S104, EX01-S106, EX01-S107. Đây là kiểm tra từ vựng, không phải đánh giá chất lượng bằng người; các loại trừ cụ thể được lưu trong audit riêng.
- Audit runtime v003 trên thư mục mẫu hiện có: 32 mẫu ban đầu, 8 phản hồi bị bộ lọc đánh dấu, còn 24 ứng viên. Chuẩn bị đủ 328 truy vấn public với sidecar mức yêu cầu hiện có đạt; mỗi truy vấn lấy 2 ví dụ cùng mức. Không chạy generation hay tính điểm thí nghiệm. Hash toàn bộ file trong `sample_dataset` trước/sau giống nhau.
- Mẫu submission chính thức là danh sách `[{"sample_id": "...", "output": {"taxonomy_error": [...], "feedback": "..."}}]`. Không có metadata trong file nộp.

Không dùng public làm tập train, làm ví dụ minh họa, fit TF-IDF hoặc tối ưu prompt. Public chỉ là truy vấn suy luận; so sánh ID/hash code giúp loại bài public vô tình nằm trong tập tham chiếu.

## Chạy trên Kaggle

1. Upload từng notebook, giữ notebook và dataset sinh viên ở chế độ private. Chọn GPU **T4 x2**, bật Internet để cài dependency và tải model công khai. Chạy ba notebook lần lượt để tránh tranh GPU.
2. Attach ZIP public hoặc thư mục giải nén. Trong cell `CONFIG`, đặt `dev_path` nếu tự phát hiện không duy nhất. Không có trường nhãn đầu vào; `labels_path` cũ bị từ chối.
3. Nếu cần theo file mức yêu cầu riêng, đặt `levels_path` tới `task3_requested_levels__DL-DNTYCN1P.json`; sidecar phải phủ đúng toàn bộ 328 ID và ghi đè mức trong dữ liệu.
4. Riêng P2: attach `sample_dataset` đã giải nén hoặc `sample_dataset.zip`. Mặc định `train_path: auto_sample_dataset` tìm tập 32 mẫu có annotation duy nhất dưới `/kaggle/input`, hoặc ZIP khi không có bản giải nén. Nếu có nhiều bản, đặt `train_path` tới đúng thư mục/ZIP. P1/P3 không cần tập tham chiếu. P2 chỉ lấy ví dụ cùng mức: hai nếu có đủ, một nếu chỉ có một, không có thì dùng zero-shot. Level 4 hiện luôn dùng zero-shot vì corpus không có mức 4. Không mượn ví dụ từ mức khác.
5. Giữ `mode: smoke` rồi Run All để chạy 10 mẫu. Sau khi kiểm tra pipeline, đặt `mode: dev` và Run All để xuất đủ 328 dự đoán. Không dùng kết quả public để chỉnh prompt.

Mặc định được đồng bộ theo cấu hình chạy đã xác nhận: Qwen/Qwen3-4B, revision bất biến `1cfa9a7` được resolve thành SHA 40 ký tự; seed 42; FP16; batch 1/GPU; 512 token đầu ra; greedy; thinking tắt. Hai model replica độc lập trên hai T4. Dùng chat template chính thức theo [model card Qwen3](https://huggingface.co/Qwen/Qwen3-4B). Giữ cùng phiên bản model, dependency và thông số giữa các phương pháp.

Hai subprocess độc lập, mỗi subprocess chỉ thấy một GPU và tải một model hoàn chỉnh. Tự dùng một worker khi chỉ có một GPU; cũng có thể đặt `gpu_ids: ['0']`. Không sharding model, không fine-tuning, không gọi LLM API ngoài.

`context_limit` bao gồm prompt và đầu ra. Không cắt đề, code, ví dụ hoặc chẩn đoán. Nếu vượt giới hạn, tăng context có chủ đích; giảm batch nếu thiếu VRAM. Đổi thông số trong một `output_root`/`shared_dir` mới. Runtime không tự chuyển phương pháp hay thay prompt để né lỗi.

Runtime chỉ chia batch khi có CUDA OutOfMemoryError thực tế, giữ nguyên thứ tự và toàn bộ input. Mẫu vẫn lỗi khi chạy riêng được lưu để retry, không làm dừng các mẫu khác. Lỗi validation không bị gọi là OOM. `recovered_oom_batch_splits` thống kê số lần chia batch đã phục hồi trong các generation được checkpoint; thời gian thử thất bại nằm trong wall time nhưng không cộng vào thời gian generation thành công. Batch mặc định hiện là 1 trên mỗi GPU.

## Khôi phục P1 hiện tại: 322 checkpoint và sáu alias lỗi

Mở `scripts/recover_task3_p1_kaggle_cell.py`, copy toàn bộ vào một cell mới trong session P1 hiện tại, rồi chạy. Không Run All, không gọi prepare/run_inference, không đổi manifest và không restart kernel. Cell dùng biến `runtime` đang được nạp và đường dẫn `/kaggle/working/task3_P1_dev/`; xuất `/kaggle/working/predictions_P1.json` sau khi đủ 328 ID hợp lệ.

Cell chỉ đọc attempt đầu tiên (`attempts[0]`) của từng file trong `failures/`, yêu cầu ended_with_eos=true, giữ nguyên feedback, canonicalize nhãn rồi gọi parse_response và validate_submission hiện có. Cả 328 mẫu được preflight trước khi tạo checkpoint mới; checkpoint đã tồn tại chỉ được đọc. Không chọn retry có vẻ tốt hơn, không xóa/sửa sáu failure logs, không tải model hoặc chạy inference. Nếu có unknown label, JSON hỏng, EOS thiếu hay nội dung nội bộ thì dừng trước khi ghi, báo lỗi để xử lý riêng. Cell có thể chạy lại mà không sửa checkpoint đã tạo. Audit riêng `recovery_normalization_*.private.json` ghi nguồn file và attempt_index=0; không thay stats/log cũ.

Runtime v005 normalize NFKC, whitespace và case trước strict taxonomy validation, chỉ thêm hai alias đã xác nhận `Lỗi syntax`/`Lỗi cú pháp` → `Lỗi biên dịch`; loại nhãn trùng theo thứ tự xuất hiện. Nhãn không biết vẫn bị từ chối, không fuzzy-match hoặc bỏ qua. P3 áp dụng cả taxonomy list và issue.label; sự kiện được ghi riêng trong `normalization_logs/`. Prompt, taxonomy chính thức và corpus RAG không đổi. Khi chạy thí nghiệm mới dùng shared_dir `task3_comparison_v005`; không dùng runtime mới để prepare lại lượt P1 đang khôi phục vì fingerprint runtime đã thay đổi.

## Kết quả, resume và lỗi

Full inference xuất `/kaggle/working/predictions_P1.json`, `predictions_P2.json`, `predictions_P3.json`. Mỗi file chỉ xuất sau khi kiểm tra đủ 328 ID, không trùng/thiếu, đúng nhãn và schema, phản hồi không rỗng và không lộ nội dung nội bộ. Kiểm tra mức độ bằng từ khóa chỉ tạo cảnh báo riêng, không chặn export. Quy định mức độ trong prompt vẫn giữ nguyên.

Smoke xuất `smoke_P1.json`, `smoke_P2.json`, `smoke_P3.json`, tuyệt đối không phải submission leaderboard. P2 dùng cả 32 mẫu làm corpus ban đầu, không chia train/validation, cho cả smoke và full inference. Bộ lọc chất lượng và loại trùng vẫn áp dụng. Chế độ validation tùy chọn chỉ còn dành cho P1/P3; P2 từ chối chế độ này để bảo đảm corpus không bị chia.

Chuẩn hóa ví dụ trong bộ nhớ: `input.taxonomy_error` cũ được chuyển vào `output.taxonomy_error`, cùng `output.feedback`. Input minh họa chứa mã nguồn đã đọc qua `code_file`, đề đã join qua `exam_id`, loại đề và mức yêu cầu; không chứa nhãn. Dữ liệu nguồn không thay đổi. `normalized_corpus.private.json` lưu đủ 32 ví dụ theo schema này. `retrieval_audit.private.json` lưu phân bố mức, nhãn thiếu, lý do lọc, số ví dụ đủ điều kiện và ID truy vấn dùng zero-shot. Hai file đều riêng tư và không phải submission.

Checkpoint ở `task3_P*_MODE/samples/`, cache P3 ở `diagnostics/`. Chạy lại cùng config/runtime/dữ liệu/dependency để resume; có thể đổi từ hai GPU sang một GPU. Fingerprint từ chối trộn kết quả khác thí nghiệm. Với lượt P1 đã có 322 checkpoint, dùng riêng cell offline bên trên để khôi phục sáu alias, không restart/prepare lại và không sửa manifest hay các checkpoint đó. Notebook/runtime v005 và shared_dir `task3_comparison_v005` dành cho lượt chạy mới; giữ nguyên thư mục so sánh của lượt cũ khi resume runtime cũ. Tham chiếu validation, log và cache đều private.

Runtime sửa JSON bị bọc fence, có lời dẫn, dấu phẩy cuối hoặc dạng Python literal mà không tạo nội dung mới. Với JSON/schema/trường bắt buộc/nội dung nội bộ không hợp lệ, mặc định thử sinh lại tối đa một lần bằng model local, cùng generation settings. Số lần gọi cơ sở là 1 cho P1/P2, 2 cho P3. Hết retry thì lưu `failures/*.json`, tiếp tục các mẫu khác. Sau lượt đầu, `failure_retry_passes=1` thử lại riêng các mẫu chưa hoàn tất. Nếu vẫn thiếu, xuất `unresolved_failures.private.json` và chặn submission; chạy lại để resume. Không tạo câu trả lời giả và không bỏ qua ID.

`warnings/*.json` ghi sample ID, requested level, generated feedback và lý do heuristic; `validation_logs/*.json` lưu lịch sử lỗi nghiêm ngặt. Nhắc tên biến, tên hàm hoặc từ khóa `return`, `cout`, `cin` không làm phản hồi bị từ chối. Bộ lọc chất lượng ví dụ RAG giữ nguyên.

Log S21 xác nhận lỗi validation, không phải OOM. Ngoài ra Transformers từng kế thừa defaults có `do_sample=True`. Runtime v004 gán generation config sạch cho model, dùng `use_model_defaults=False` và `do_sample=False` ngay tại lời gọi generate; sampling parameters để None. Hành vi kế thừa được đối chiếu với [source Transformers 4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/generation/utils.py). `worker_*_startup.json` ghi config hiệu lực sau khi kiểm tra greedy, PyTorch/CUDA availability, GPU visibility, model device, dtype và bộ nhớ; memory hiện tại/cực đại còn có trong worker stats. Không dùng device_map auto. Seed 42, 512 token và thinking=False giữ nguyên.

`runtime_stats.json` lưu thời gian phiên gồm load model, thời gian generation theo GPU, token, số lượt sinh theo mẫu, retry, dependency và bộ nhớ GPU cực đại. Tổng có cả phần đã cache; không tính phần đang chạy dở chưa checkpoint, cài package hay chuẩn bị dữ liệu. Manifest lưu model revision, Git revision nếu có, hash runtime, split và thông tin nguồn dữ liệu.

## Giới hạn kiểm chứng

Máy local không có CUDA. Kiểm thử CPU dùng model giả để kiểm tra luồng, batching, resume, retry và export; không chứng minh Qwen3 chạy vừa VRAM trên T4 hoặc chất lượng dự đoán. Chưa có số đo Level Compliance/Diagnostic Accuracy hay runtime thật của model. Kiểm tra từ vựng và khóa nhãn P3 không thay thế đánh giá ngữ nghĩa về tính nhất quán giữa nhãn và phản hồi.

Lệnh kiểm tra: `python -m pytest`, `python -m ruff check . --exclude tmp`, và chạy builder để kiểm tra ba notebook có runtime đồng nhất. Notebook JSON và từng code cell được kiểm tra cú pháp; schema nbformat được kiểm tra nếu có dependency tùy chọn.

Kết quả local runtime v004 ngày 2026-11-25 (Python 3.11): toàn bộ pytest đạt **221 passed, 1 skipped, 15 subtests passed**. Một test schema nbformat được bỏ qua vì chưa cài dependency tùy chọn; JSON, cú pháp code, runtime nhúng và output trống của cả ba notebook đã được kiểm tra. Ruff đạt cho `src tasks scripts tests` và riêng ba notebook mới. Lint toàn repo vẫn có lỗi trong các notebook cũ ngoài phạm vi thay đổi. Regression tests kiểm tra cảnh báo lexical không gây retry/chặn export, xử lý lỗi riêng từng mẫu cho P1/P2/P3, lượt retry cuối, resume giữ checkpoint, completeness 328 ID và generation config greedy không kế thừa sampling defaults. Input thật của S21 đã được đọc và kiểm tra; phản hồi gây lỗi cũ không có trong log nên câu regression là minh họa tái hiện cùng loại false positive, không phải output được khôi phục.

Máy kiểm thử hiện có PyTorch 2.14.1+cpu, CUDA=False, GPU count=0; chưa chạy smoke thật trên hai T4. Tests sử dụng model giả chạy luồng smoke 10 mẫu qua hai worker; không được xem là kết quả GPU hoặc điểm chất lượng.

Kiểm tra v005: full suite **238 passed, 1 skipped, 15 subtests passed**; cell recovery cuối cùng có **16 tests passed**, bao gồm JSON bọc fence, tình huống 322+6, feedback giữ nguyên, unknown label, lỗi EOS/nội dung nội bộ, và chạy lại không ghi đè checkpoint/log. Ruff đạt cho mã Python và ba notebook mới. Các test worker P1/P2/P3 xác nhận normalization không gây thêm lượt inference và có log riêng. Dữ liệu recovery trong test là fixture đúng schema; sáu file failure thực tế nằm trong session Kaggle nên chưa được thực thi tại local.
