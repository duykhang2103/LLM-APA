# BRIEF HỌP NHÓM ĐỒ ÁN LARGE LANGUAGE MODEL

> **Current scope (2026-10-04):** four-week course project, about two weeks for experiments. Follow the runnable CLI/notebook guide in `docs/project/RUN_EXPERIMENTS.md` and current `PROJECT_PLAN.md`. This document retains historical setup/design/brainstorming; its old placeholder status, hardware/test observations, broad method ladders, and schedules do not establish current execution. P0/P1/F0 are implemented; actual Qwen GPU smoke is still required. SemIf/F1, diagnostic stages, self-consistency and threshold/calibration experiments are not prerequisites. Do not invent EX01 per-question-to-rubric scoring.

## Challenge: Chấm điểm và phản hồi tự động cho bài lập trình của sinh viên

Nguồn đã đọc:

- [Yêu cầu đồ án](https://docs.google.com/document/d/1Dr3g-JbZ1SsucZ0wsdz85IA0bRksxYU7/edit)
- [Sample dataset](https://drive.google.com/file/d/1iqkw6ZWUV6-I6rsVQdqHBOhiRgRkk_UZ/view?usp=sharing)

> Mục tiêu của brief: giúp nhóm đi vào buổi họp với cùng một cách hiểu về đề, có sẵn các hướng kỹ thuật để tranh luận, và kết thúc buổi họp bằng các quyết định cụ thể thay vì chỉ dừng ở việc “chọn model”.

## 1. Đề bài thực sự yêu cầu gì?

Nhóm cần xây một pipeline LLM tự động xử lý bài nộp C++ cho hai loại đề: `multi_problem` và `single_problem`. Hệ thống phải giải quyết ba task:

| Task | Input chính | Output | Metric chính | Điểm khó cần chú ý |
|---|---|---|---|---|
| Task 1 — Chấm rubric | Đề bài, code, tùy chọn compile log/test report | 6 điểm thành phần và tổng 0–10 | QWK trên tổng điểm | Phải áp dụng chính sách chấm, nhất là quan hệ tiên quyết giữa các câu |
| Task 2 — Phân loại lỗi | Giống Task 1 | Tập con của 10 nhãn lỗi; tập rỗng hợp lệ | Macro-F1 | Nhãn mất cân bằng và có thể có nhãn gần suy biến |
| Task 3 — Sinh phản hồi | Đề, code, nhãn lỗi cho sẵn, mức phản hồi | Phản hồi tiếng Việt | Level compliance và độ chính xác chẩn đoán | Level 1/2 không được vô tình tiết lộ cách sửa hoặc code |

### Các ràng buộc kỹ thuật bắt buộc

1. So sánh tối thiểu hai hướng tiếp cận.
2. Có ít nhất một hướng prompting trên LLM: zero-shot, few-shot, chain-of-thought, self-consistency, RAG hoặc kết hợp.
3. Có ít nhất một mô hình open-weight được fine-tune; tài liệu khuyến nghị mô hình không quá 10B tham số. Mô hình phải chạy cục bộ và nhóm phải báo cáo phần cứng.
4. Pipeline suy luận phải tự động hoàn toàn, không sửa tay từng bài.
5. Cố định seed, phiên bản model và môi trường để có thể tái lập.
6. Nếu dùng API thương mại thì phải báo cáo chi phí.
7. Không được dùng trường `feedback` làm input cho Task 1 hoặc Task 2.

### Cơ cấu điểm cần chi phối chiến lược

| Thành phần | Tỷ trọng |
|---|---:|
| Private leaderboard | 30% |
| Phương pháp và thiết kế thí nghiệm | 25% |
| Ablation và phân tích lỗi | 15% |
| Báo cáo và trình bày | 20% |
| Khả năng tái lập và chất lượng mã nguồn | 10% |

Như vậy, 70% số điểm không phụ thuộc trực tiếp vào thứ hạng leaderboard. Mục tiêu hợp lý không phải chỉ là “model mạnh nhất”, mà là một hệ thống có giả thuyết rõ, thí nghiệm kiểm chứng được, chạy lại được bằng một lệnh và giải thích được vì sao đúng/sai.

### Sản phẩm phải nộp

- Repository riêng tư, hướng dẫn cài đặt và script tái lập kết quả tốt nhất bằng một lệnh.
- Báo cáo 10–15 trang: EDA, phương pháp, ablation, phân tích lỗi, chi phí, khả năng triển khai và hạn chế.
- Model card ngắn: dữ liệu huấn luyện, giới hạn và rủi ro khi dùng để chấm thật.
- Slide, thuyết trình 15 phút và hỏi đáp 10 phút.
- `predictions.json` đúng schema cho tập test riêng tư.
- Bảng đóng góp của từng thành viên, có xác nhận của cả nhóm.

## 2. Hướng làm đề xuất để nhóm tranh luận

### Ý tưởng trung tâm: Shared Diagnostic Backbone

Thay vì làm ba hệ thống rời, xây một tầng phân tích bài nộp dùng chung:

```text
exam metadata + statement + code + compile log/test report
                         |
                         v
              Submission Diagnostic Record
  - compile evidence
  - I/O evidence
  - per-problem status
  - failed/public tests
  - suspected error types
  - evidence spans in code
                         |
            +------------+------------+
            |            |            |
            v            v            v
       Rubric head   Taxonomy head  Feedback renderer
        (Task 1)       (Task 2)       (Task 3)
```

Lợi ích:

- Một nguồn bằng chứng chung giúp ba task nhất quán.
- Có thể ablation từng loại tín hiệu: code, compile log, test report, policy của đề.
- Dễ tách phần “suy luận nội bộ” khỏi output JSON bắt buộc.
- Có đường dẫn rõ để so sánh prompting với fine-tuning.

## 3. Hai hướng tiếp cận tối thiểu

### Hướng A — Prompting + retrieval + ràng buộc đầu ra

Baseline đầu tiên có thể là zero-shot với schema JSON. Sau đó phát triển thành:

1. Chọn few-shot theo `exam_type`, đề, nhãn lỗi hoặc độ giống của code.
2. RAG chỉ trên tập train: truy xuất các bài tương tự cùng nhãn/rubric, tuyệt đối không lấy `feedback` làm ngữ cảnh cho Task 1/2.
3. Yêu cầu model tạo `Diagnostic Record` có dẫn chứng, sau đó mới ánh xạ sang output.
4. Dùng self-consistency ở những trường hợp không chắc chắn; bỏ phiếu theo từng chiều rubric hoặc từng nhãn thay vì bỏ phiếu cả chuỗi JSON.
5. Post-processing xác định: ép miền giá trị, tính tổng điểm từ sáu chiều và áp dụng policy tiên quyết.

Điểm mạnh: nhanh có baseline, dễ giải thích và dễ tạo ablation. Điểm yếu: chi phí suy luận, độ dao động và dễ bị prompt dài khi code lớn.

### Hướng B — Fine-tune open-weight bằng LoRA/QLoRA

Fine-tune một model code-capable không quá 10B theo dạng instruction-to-structured-output. Hai phương án để brainstorm:

- Một model đa nhiệm, thêm task token như `<GRADING>`, `<ERRORS>`, `<FEEDBACK_L2>`.
- Một backbone chung nhưng adapter riêng cho Task 1/2/3.

Đề xuất thực dụng: bắt đầu bằng adapter riêng để dễ debug, sau khi có baseline mới thử multi-task để kiểm tra transfer giữa các task. Với Task 3, đưa `target_feedback_level` thành điều kiện bắt buộc và bổ sung ví dụ tương phản giữa các level nếu dữ liệu cho phép.

Điểm mạnh: chạy cục bộ, chi phí suy luận thấp hơn và đáp ứng đúng yêu cầu môn học. Điểm yếu: dễ overfit khi dữ liệu nhỏ, cần quản lý phần cứng và thí nghiệm cẩn thận.

## 4. Ý tưởng riêng cho từng task

### Task 1 — Chấm điểm theo rubric

1. **Chấm theo hai tầng:** model dự đoán bằng chứng/per-problem status trước, sau đó bộ luật tính rubric và tổng điểm.
2. **Policy engine cho đề nhiều câu:** nếu câu tiên quyết không đạt, khóa điểm các câu phụ thuộc. Đây nên là logic xác định, không giao hoàn toàn cho LLM.
3. **Tách tín hiệu quan sát và ground truth:** compile log chỉ là bằng chứng phụ. Không ánh xạ máy móc `compile_log fail` thành `compilable = 0`.
4. **Ordinal modeling:** tổng điểm có thứ tự, vì vậy có thể thử dự đoán theo ordinal classification thay vì 11 lớp độc lập.
5. **Calibration:** cho model xuất độ tin cậy nội bộ; dùng độ tin cậy để quyết định khi nào chạy self-consistency, không dùng để can thiệp thủ công.

### Task 2 — Phân loại lỗi đa nhãn

1. Dự đoán từng nhãn dưới dạng bài toán nhị phân với mô tả và ví dụ của riêng nhãn đó.
2. Tune threshold riêng cho từng nhãn trên validation thay vì dùng cố định 0.5.
3. So sánh BCE thường, class-weighted BCE và focal loss để xử lý nhãn hiếm.
4. Thử hierarchy mềm: trước hết xác định nhóm compile/I-O/semantic/quality, sau đó mới chọn nhãn cụ thể.
5. Kiểm tra xung đột giữa nhãn và bằng chứng, ví dụ “Lỗi biên dịch” nhưng log rỗng; không tự động xóa vì log có thể lệch môi trường.

### Task 3 — Phản hồi có kiểm soát mức độ

1. **Plan-then-render:** tạo kế hoạch nội dung không hiển thị, rồi render theo level.
2. **Level-specific template/policy:** mỗi level có danh sách nội dung được phép và bị cấm.
3. **Compliance checker:** sau khi sinh, kiểm tra code block, dòng code, từ khóa chỉ cách sửa và mức độ chi tiết. Nếu vi phạm thì tự động regenerate.
4. **Contrastive level training:** cùng một chẩn đoán nhưng viết bốn phiên bản L1–L4 để model học ranh giới mức độ. Nếu tạo dữ liệu tổng hợp, phải tách khỏi validation và báo cáo rõ nguồn sinh.
5. **Đánh giá hai trục riêng:** correctness và compliance; không gộp quá sớm để thấy model tốt nội dung nhưng kém sư phạm hay ngược lại.

## 5. Ma trận thí nghiệm và ablation nên ưu tiên

Không nên chạy mọi tổ hợp. Chọn 5–7 thí nghiệm trả lời các câu hỏi quan trọng nhất:

| Câu hỏi nghiên cứu | So sánh đề xuất | Metric |
|---|---|---|
| Tín hiệu phụ có thật sự giúp không? | code only vs +compile log vs +test report | QWK, macro-F1 |
| Policy của đề nhiều câu có giá trị không? | LLM thuần vs +deterministic prerequisite engine | QWK và lỗi riêng EX01 |
| Retrieval có giúp hơn few-shot cố định? | zero-shot vs fixed few-shot vs similarity RAG | Theo từng task |
| Ràng buộc output có giảm lỗi không? | free generation vs JSON schema + validator | invalid rate và metric task |
| Xử lý mất cân bằng nào tốt hơn? | threshold 0.5 vs threshold theo nhãn vs class weighting | macro-F1, per-label F1 |
| Kiểm soát feedback hiệu quả ra sao? | one-pass vs plan-render vs +compliance checker | violation rate, correctness |
| Fine-tune có hơn prompting không? | best prompt vs LoRA/QLoRA cùng input budget | metric, latency, VRAM, chi phí |

Mỗi thí nghiệm phải lưu: commit/config ID, seed, model version, prompt version, split, metric, latency, chi phí và file prediction.

## 6. Những gì sample dataset đang gợi ý

Các quan sát sau chỉ áp dụng cho bộ mẫu 32 bài, chưa được xem là kết luận về tập chính thức:

- 15 bài `EX01` và 17 bài `EX02`.
- Điểm tổng chỉ xuất hiện ở 0, 1, 2, 3, 6, 7, 8, 9, 10; không có điểm 4 hoặc 5. Phân bố có dấu hiệu hai cụm thấp/cao.
- Task 2 có 4 bài không mang nhãn lỗi nào. Nhãn `Lỗi edge case` không xuất hiện trong bộ mẫu; các nhãn còn lại lệch mạnh, từ 1 đến 12 lần xuất hiện.
- Task 3 có 4 mẫu Level 1, 18 mẫu Level 2, 10 mẫu Level 3 và không có Level 4. Toàn bộ EX01 trong sample đều ở Level 2, tạo nguy cơ model học tắt từ `exam_id` sang level/style.
- README cảnh báo có 9/121 trường hợp trong dữ liệu nguồn mà biên dịch cục bộ thất bại nhưng giảng viên vẫn cho `compilable = 1`, chủ yếu do khác biệt MSVC/C++11. Đây là dấu hiệu rõ rằng compile log không thể được coi là nhãn tuyệt đối.

### Các hướng săn điểm thưởng +10%

1. **Nhãn suy biến/hiếm:** đo support, per-label F1 và độ ổn định bootstrap; đề xuất gộp nhãn, bổ sung dữ liệu hoặc metric có confidence interval.
2. **Mâu thuẫn compile log và rubric:** định lượng theo exam type, loại lỗi compiler và môi trường; đề xuất chuẩn hóa toolchain hoặc tách `compiles_on_reference_env` khỏi `teacher_compilable`.
3. **Leakage do duplicate/near-duplicate code:** đo clone similarity giữa train/dev; nếu cùng template hoặc code gần giống nằm ở hai split thì leaderboard có thể quá lạc quan.
4. **Confounding giữa exam và feedback level:** đo phân bố `P(level | exam_id)`; đánh giá chéo exam để xem model học level thật hay học mã đề.
5. **Rubric/taxonomy/feedback không nhất quán:** kiểm tra ví dụ logic = 4 nhưng có nhãn lỗi logic, hoặc phản hồi nêu lỗi không có trong taxonomy.
6. **Metric Task 3 chưa đủ định lượng:** kiểm tra độ đồng thuận giữa người chấm hoặc giữa các judge; đề xuất rubric tách factuality, actionability và level compliance.
7. **QWK và phân bố điểm hổng:** báo cáo thêm MAE theo vùng điểm, confusion matrix và bootstrap CI để tránh một metric che khuất lỗi ở nhóm hiếm.

Mọi phát hiện cần có bằng chứng định lượng, không chỉ nêu cảm nhận.

## 7. Checklist chuẩn bị trước buổi meeting

### Mỗi thành viên chuẩn bị

- Đọc yêu cầu và README của sample dataset.
- Chạy thử parser trên 32 bài; đọc ít nhất một bài EX01 và một bài EX02.
- Ghi lại năng lực/kinh nghiệm có thể đảm nhận: data, prompting, fine-tuning, backend/pipeline, MLOps, viết báo cáo, thuyết trình.
- Báo phần cứng có thể dùng: GPU, VRAM, RAM, hệ điều hành, số giờ máy có thể chạy.
- Báo thời gian có thể đóng góp mỗi tuần trong 8 tuần.
- Mang tới một giả thuyết muốn kiểm chứng, không chỉ một tên model muốn thử.

### Nhóm chuẩn bị chung

- Private repository; thêm dữ liệu, checkpoint, output và secret vào `.gitignore`.
- Data manifest và checksum để mọi người dùng đúng phiên bản nhưng không commit dữ liệu.
- Một script evaluator cục bộ cho QWK, MAE, exact match, macro/micro-F1 và compliance sơ bộ.
- Schema validator cho `predictions.json` trước khi submit; vì file lỗi vẫn bị trừ quota.
- Experiment registry tối thiểu bằng CSV/JSON/W&B/MLflow, tùy khả năng nhóm.
- Bảng theo dõi leaderboard gồm ngày, commit, config, public score và nhận xét; tối đa 2 lượt/ngày nên mỗi lượt phải có giả thuyết.
- Chính sách dùng API và bảo mật dữ liệu được cả nhóm thống nhất.

## 8. Câu hỏi phải chốt với giảng viên

1. Nhóm bắt buộc nộp cả ba task hay có thể chọn task?
2. “Khuyến nghị ≤10B” là khuyến nghị hay giới hạn cứng? “Chạy cục bộ” yêu cầu cả huấn luyện và suy luận hay chỉ suy luận?
3. Tập train/dev/test được split theo bài nộp, sinh viên, đề bài hay học kỳ? Có nguy cơ cùng template/đề nằm ở nhiều split không?
4. Task 3 được chấm bằng người, rule hay LLM judge? Định nghĩa chính xác của một vi phạm Level 1/2 là gì?
5. `total_score` bắt buộc bằng tổng sáu chiều hay có quy tắc làm tròn/ngoại lệ? Chính sách tiên quyết được cung cấp ở dạng máy đọc hay phải trích từ statement?
6. Được dùng dữ liệu tổng hợp, model teacher hoặc dữ liệu code công khai để fine-tune không? Nếu được, phải khai báo ở mức nào?
7. Được gửi code sinh viên tới API thương mại không? Nếu được, nhà cung cấp nào và cấu hình lưu trữ/retention nào được chấp nhận?
8. Khi compile log mâu thuẫn với nhãn giảng viên, ground truth nào được ưu tiên? Có cung cấp toolchain chuẩn không?
9. Script validate chính thức và schema output cho từng task sẽ được phát khi nào?
10. Có hạn chế về ensemble, số lần gọi model cho một bài hoặc chi phí suy luận không?

Điểm số 7 đặc biệt quan trọng: tài liệu cho phép dùng API nhưng đồng thời cấm phát tán dữ liệu ra ngoài lớp. Không nên tự giả định việc gửi code lên API là hợp lệ; cần xác nhận rõ và lưu quyết định.

## 9. Agenda meeting đề xuất — 75 phút

1. **0–10 phút — Đồng bộ yêu cầu:** ba task, ràng buộc bắt buộc, cơ cấu điểm và bảo mật.
2. **10–20 phút — Nguồn lực:** kỹ năng thành viên, GPU/VRAM, ngân sách API, thời gian mỗi tuần.
3. **20–40 phút — Brainstorm kỹ thuật:** tranh luận Shared Diagnostic Backbone, hướng prompting và hướng fine-tune.
4. **40–52 phút — Chọn câu hỏi nghiên cứu:** chốt 3–5 giả thuyết và ablation ưu tiên.
5. **52–62 phút — Phân công:** owner cho data/evaluation, Task 1, Task 2, Task 3, fine-tuning, reproducibility/report.
6. **62–70 phút — Kế hoạch 8 tuần:** milestone, Definition of Done và nhịp review.
7. **70–75 phút — Chốt hành động tuần đầu:** ai làm gì, deadline, đầu ra kiểm chứng được.

### Các quyết định buổi họp phải tạo ra

- Kiến trúc baseline v0.
- Model/API dùng cho baseline prompting và 2–3 ứng viên open-weight sẽ benchmark theo phần cứng.
- Cách chia train/validation cục bộ và nguyên tắc chống leakage.
- 3–5 giả thuyết thí nghiệm ưu tiên.
- Owner và reviewer cho từng workstream.
- Công cụ quản lý experiment và chuẩn đặt tên run.
- Nguyên tắc bảo mật/API.
- Milestone tuần 1 và tuần 2.

## 10. Phân công gợi ý cho nhóm 5–7 người

| Vai trò | Trách nhiệm chính |
|---|---|
| Data & Evaluation owner | Loader, schema, EDA, split, metric, leakage checks |
| Task 1 owner | Rubric prompting/modeling, policy engine, QWK/error analysis |
| Task 2 owner | Multi-label modeling, imbalance, threshold tuning |
| Task 3 owner | Controlled feedback, compliance checker, evaluation |
| Fine-tuning/MLOps owner | Model benchmark, LoRA/QLoRA, training, checkpoint, hardware report |
| Reproducibility/Integration owner | Pipeline một lệnh, config, seed, packaging prediction |
| Report/Presentation owner | Báo cáo, model card, slide; thu thập bằng chứng từ đầu |

Nếu nhóm chỉ có 5 người, gộp Data với Integration và gộp Report với một Task owner. Mọi hạng mục quan trọng nên có một owner và một reviewer; không chia theo kiểu “tất cả cùng chịu trách nhiệm”.

## 11. Kế hoạch 8 tuần đề xuất

| Tuần | Mục tiêu | Đầu ra tối thiểu |
|---|---|---|
| 1 | Hiểu dữ liệu và dựng khung tái lập | Loader, validator, EDA v1, local split, metric, risk log |
| 2 | Baseline end-to-end | Zero-shot/few-shot chạy đủ 3 task, tạo đúng `predictions.json` |
| 3 | Prompt/RAG và policy engine | Baseline A hoàn chỉnh, kết quả và error buckets |
| 4 | Fine-tune pilot | Chọn model theo phần cứng, LoRA pilot, hardware/cost log |
| 5 | Fine-tune chính + Task 3 control | Baseline B hoàn chỉnh, compliance checker |
| 6 | Ablation và phân tích lỗi | Bảng ablation, per-label/per-exam analysis, chọn ứng viên cuối |
| 7 | Freeze hệ thống | One-command reproduction, model card, draft báo cáo và slide |
| 8 | Final QA và trình bày | Re-run sạch, predictions cuối, rehearsal 15+10 phút |

Nguyên tắc: hết tuần 5 phải có hai hướng hợp lệ chạy end-to-end. Tuần 6–8 dành cho bằng chứng, sửa lỗi và truyền đạt; không nên tiếp tục đổi kiến trúc lớn nếu không có lý do mạnh.

## 12. Risk register ban đầu

| Rủi ro | Tác động | Giảm thiểu |
|---|---|---|
| Dữ liệu nhỏ/mất cân bằng | Overfit, macro-F1 thấp | Stratified/group split, per-label threshold, bootstrap CI |
| Leakage qua feedback hoặc duplicate | Kết quả ảo, vi phạm đề | Field whitelist, clone detection, group split |
| Compile log lệch môi trường | Chấm sai compilable | Xem log là evidence, chuẩn hóa toolchain, ablation |
| Feedback L1/L2 lộ đáp án | Vi phạm compliance | Policy rõ, checker, regenerate, test set chuyên biệt |
| Dev leaderboard bị overfit | Private score giảm | CV cục bộ, submission log, chỉ submit theo giả thuyết |
| Không đủ GPU/VRAM | Trễ fine-tune | Benchmark sớm, QLoRA, giới hạn context/batch, fallback model nhỏ hơn |
| Pipeline không tái lập | Mất 10% và khó debug | Config hóa, seed, container/env lock, smoke test một lệnh |
| Dữ liệu gửi ra API trái chính sách | Rủi ro học thuật/bảo mật | Xác nhận giảng viên, chọn no-retention hoặc ưu tiên local |

## 13. Đề xuất hành động ngay sau meeting

Trong 48 giờ đầu:

1. Data owner hoàn thiện data inventory, schema và EDA v1.
2. Integration owner dựng CLI thống nhất: `prepare`, `train`, `predict`, `evaluate`, `package`.
3. Task owners viết baseline prompt/schema và bộ 5–10 case kiểm thử thủ công.
4. Fine-tuning owner benchmark inference 2–3 model ứng viên trên phần cứng thật trước khi chọn model.
5. Nhóm chốt policy bảo mật và ghi lại quyết định về API.
6. Tạo experiment log và quy ước ID trước run đầu tiên.

### Definition of Done cho tuần 1

- Mọi thành viên clone repo và chạy được pipeline smoke test.
- Đọc code đúng từ `code_file`; không đưa feedback vào Task 1/2.
- Metric chạy được trên sample dataset.
- Có báo cáo EDA ngắn và danh sách ít nhất năm giả thuyết dữ liệu.
- Có local split chống leakage theo quyết định của nhóm.
- Có baseline stub xuất đúng schema cho cả ba task.
- Có bảng nguồn lực phần cứng và quyết định model thử nghiệm đầu tiên.

## 14. Câu mở đầu gợi ý cho buổi brainstorm

> Mục tiêu của nhóm không chỉ là lên leaderboard. Chúng ta cần chứng minh được ba điều: hệ thống chấm nhất quán với policy của đề, nhận diện được nhãn hiếm thay vì chỉ đoán lớp phổ biến, và sinh phản hồi đúng mức độ mà không lộ lời giải. Tôi đề xuất dùng một tầng phân tích bài nộp chung, sau đó so sánh hướng prompting/RAG với hướng fine-tune open-weight. Hôm nay nhóm cần chốt các giả thuyết sẽ kiểm chứng, nguồn lực phần cứng, cách chống leakage và owner cho từng đầu ra.

