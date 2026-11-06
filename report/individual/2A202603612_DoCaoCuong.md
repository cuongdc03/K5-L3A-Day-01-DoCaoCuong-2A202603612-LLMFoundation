# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603612
- **Họ và tên:** Đỗ Cao Cường
- **SEED** (in ở ô đầu notebook): 7582

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Ô "Báo cáo cuối": 15/15 kiểm tra đạt, đủ 10 dự đoán.

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (lần chạy cuối). Hàng LLM chỉ đo trên SUB 40 tin, nên so chất lượng bằng cột F1 SUB 40.

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|
| 1. BoW + LogReg (tự code) | 0.898 | 0.788 | 0.04 | 0 |
| 2. BoW + MLP | 0.940 | 0.919 | 0.10 | 0 |
| 3a. Encoder + kNN | 0.919 | 0.947 | 37.9 | 0 |
| 3b. Encoder + LogReg | 0.941 | 0.947 | 14.6 | 0 |
| 3c. Encoder + MLP head | 0.941 | 0.947 | 23.5 | 0 |
| 4a. LLM nhỏ zero-shot | (chỉ SUB) | 0.923 | 1832 | 0.006 |
| 4b. LLM lớn zero-shot | (chỉ SUB) | 0.947 | 1535 | 0.065 |
| 4c. LLM nhỏ few-shot | (chỉ SUB) | 0.947 | 1526 | 0.015 |

Trên đường Pareto: BoW + LogReg, BoW + MLP, Encoder + LogReg (3b). Các cách còn lại bị lấn át. SUB chỉ 40 tin nên chênh dưới ~0.05 F1 (khoảng 1 tin) coi như hoà.

Ghi chú: lúc đầu cấu hình lab `max_tokens=3` khiến model trả về rỗng (`finish_reason="length"`, `content=''`), mọi nhãn thành ham và F1 = 0. Tôi đã quét ngân sách token (3 → rỗng, từ 5 → 'spam') và dùng 16 token cho các lần chạy cuối.

## 3. Dự đoán và thực tế

1. **BoW báo nhầm ham chứa `free`/`win`:** tôi đoán BoW sẽ sai cả hai tin, thực tế BoW đúng cả hai. Logistic regression cộng bằng chứng của mọi từ trong tin, các từ ham ("gym", "tomorrow", "lol"…) kéo tổng điểm về ham nên từ khoá spam đứng một mình không đủ đổi nhãn (chưa kiểm chứng bằng cách in trọng số).
2. **Độ tụt trên dữ liệu khó:** tôi đánh giá thấp mức tụt. F1 của BoW tụt từ ~0.8–0.9 xuống 0.33–0.38, của encoder xuống 0.64–0.76. Tập test thường cùng phân phối với tập train nên quá lạc quan.
3. **LLM (`hard_eval`):** tôi đoán LLM đạt 80–90% và yếu nhất ở prompt injection; thực tế LLM nhỏ đúng 50/50 (cả prompt thường lẫn prompt phòng thủ), kể cả nhóm injection. Tôi nghĩ LLM sẽ bị "ignore all previous instructions" lừa, có thể do model hiện đại tách tốt dữ liệu và chỉ dẫn (chưa kiểm chứng). Ngoài ra dự đoán "terra F1 cao hơn luna" sai: 0.947 so với 0.923, chỉ chênh 1 tin, dù đắt gấp 10 lần.

## 4. Phân tích 50 case khó

F1 trên 50 case khó (so với F1 SUB thường): BoW + MLP 0.919 → 0.378 (tụt nhiều nhất, −0.54), BoW + LogReg 0.788 → 0.333, Encoder + LogReg 0.947 → 0.640, Encoder + MLP head 0.947 → 0.680, Encoder + kNN 0.947 → 0.759 (bền nhất trong nhóm không dùng LLM). LLM 0.923 → 1.000 (không tụt).

Hai case cụ thể (số % là tỉ lệ đúng của cả nhóm chứa case đó):
- **#21 "Hi Mum, I dropped my phone in the toilet… send £200"** (Thao túng): BoW đúng 12–25% ở nhóm này. Tin giả làm người thân không chứa từ khoá spam nào, mà BoW chỉ nhìn token rời rạc nên (suy luận, chưa in trọng số) không có tín hiệu spam. Encoder + LogReg cũng chỉ đúng 38% nhóm này, kNN 100%, LLM 100% vì đọc được ý định của tin.
- **#29 "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…"** (Né từ khoá): ở nhóm này BoW + MLP đúng 0%, BoW + LogReg 17%, vì các token như `c0ngr4tulat10ns` nằm ngoài vocab và bị bỏ qua. Encoder tách thành mảnh gần "congratulations", "gift", "card" nên đúng 83–100%.

Một cơ chế khác: BoW đạt 100% ở nhóm "Nói về spam" nhưng chỉ vì nó đoán ham cho gần mọi tin (accuracy tổng 52–54%), không phải vì hiểu.

## 5. Thử thách cá nhân

**Đề #5 (SEED 7582 % 6 = 4):** viết 4 tin của riêng mình (2 spam né từ khoá, 2 ham chứa `free`/`win`), so BoW với Encoder + LogReg.

Kết quả: BoW đúng 2/4, Encoder + LogReg đúng 3/4.
- Tin 1 (spam viết kiểu `cl@im`, `g1ft`): BoW sai vì token lạ nằm ngoài vocab; encoder tách thành mảnh gần "gift", "card", "claim" nên vẫn đúng.
- Tin 2 (spam giọng tự nhiên, không từ khoá): cả hai đều sai (đúng nhóm yếu nhất của cả BoW và encoder ở bài stress test).
- Tin 3, 4 (ham chứa `free`, `win`): cả hai đều đúng, vì các từ ham khác kéo điểm về ham.

Chỉ 4 tin nên đây chỉ là minh hoạ, không đủ kết luận chất lượng chung.

## 6. Phương án cho sản phẩm thật

**Chọn: Encoder (MiniLM) + LogReg (3b)** — F1 test 0.941, p95 ~14.6 ms (< 50 ms), chi phí gần 0, ~520 tin/s khi chạy theo lô so với nhu cầu ~116 tin/s (10 triệu/ngày). BoW + MLP cũng đạt F1 0.940 và nhanh hơn (0.10 ms) nhưng chỉ đạt F1 0.378 trên 50 case khó (3b: 0.640).

**Đánh đổi:** chậm hơn BoW ~15 ms; vẫn sót spam giọng tự nhiên (đúng chỉ 20% nhóm này). LLM không đạt SLA (p95 ~1,5–1,8 s) và tốn tiền: ~$1.800 (4a), ~$4.500 (4c), ~$19.500 (4b) mỗi tháng cho 300 triệu tin; để đạt 116 tin/s cần ~130–150 lời gọi song song.

**Đề xuất lai (router):** chuẩn hoá đầu vào; encoder + LogReg cho mọi tin; p < 0.3 → ham, p > 0.7 → spam; chỉ p ∈ [0.3, 0.7] (ước 5–10% tin) chuyển sang LLM nhỏ với prompt phòng thủ, chạy bất đồng bộ. Chi phí LLM giảm còn ~$90–180/tháng, và LLM chỉ xử lý ca khó — nơi nó đạt 100% trên 50 case khó. LLM không phải lớp kiểm tra duy nhất.

## 7. AI log và điều đã học

Trung thực: code các hàm TODO 2.1–4.4 và bản nháp nhiều câu trả lời do AI (Claude Code) viết.

| Phần | Hỏi AI gì | AI viết gì | Tôi sửa / kiểm chứng gì |
|---|---|---|---|
| 2.1–2.2 | `build_mlp`, `train_mlp` | MLP và vòng huấn luyện | Chạy `_run_check`, kiểm tra loss giảm |
| 3.1–3.3 | `cosine_similarity`, `top_k_similar`, `knn_predict` | Code vectorized | Chạy test của lab |
| 4.1–4.4 | `call_openai`, prompt, `parse_label`, `llm_classify_batch` | Code từng hàm; sửa `call_openai` tạo client trong hàm để mock hoạt động | Chạy test mock; quét `max_tokens` |
| 1.4 | F1 ra 2.0 | Chỉ ra mẫu số `precision * recall` sai | Sửa thành `precision + recall`, chạy lại |
| Điều tra F1 LLM = 0 | Vì sao F1 = 0 | `max_tokens=3` làm content rỗng | Chạy quét token và chạy lại |
| Thử thách Đề #5 | 4 tin tự nghĩ và code so sánh | Soạn tin và code ô thử thách | Chạy ô, so với dự đoán (đề ghi không dùng AI, tôi ghi lại trung thực) |
| Rà soát notebook | Kiểm tra toàn bộ code và văn bản | Phát hiện biến `b` bị ghi đè trong ô thử thách, ô báo cáo cuối bị mất, số liệu cũ | Chạy lại toàn bộ, đối chiếu từng số với output |

**Điều tôi hiểu ra:** ⏳ TODO — viết 2–3 câu bằng lời của bạn (gợi ý: F1 tốt hơn accuracy khi lệch lớp; cấu hình sai như `max_tokens` làm mô hình hỏng im lặng; tập test thường quá lạc quan so với dữ liệu khó; model lớn đắt gấp 10 lần chưa chắc tốt hơn).
