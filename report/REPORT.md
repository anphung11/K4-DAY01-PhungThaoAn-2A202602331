# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): class_id: 468; class_name: "cab" ; rank: 1 ; score: 0.510915 ; taxonomy_name: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào? Record này dùng một nhãn duy nhất ("cab") để đại diện cho toàn bộ bức ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Danh sách các lớp này được định nghĩa bởi nhóm tác giả tạo ra bộ dữ liệu ImageNet (như hiển thị ở trường taxonomy_name: "ImageNet-1K"). Checkpoint yolo11n-cls.pt được huấn luyện trên bộ dữ liệu này, nên nó bị giới hạn dự đoán cấu trúc thế giới thông qua lăng kính của đúng 1.000 nhãn mà nhóm tác giả ImageNet đã quy định từ trước (trong đó lớp số 468 là "cab").
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? Tên taxonomy ("ImageNet-1K"): Cho biết ta đang dùng "từ điển" nào. Cùng là xe taxi nhưng ở bộ dữ liệu COCO hay OpenImages có thể nó mang ID khác hoặc tên gọi khác.

ID (468): Là mã định danh duy nhất dành cho máy tính xử lý. Dùng ID giúp tránh các lỗi do con người (như gõ sai chính tả, khác biệt ngôn ngữ, viết hoa/thường).

Tên lớp ("cab"): Dành cho con người (kỹ sư, người gán nhãn) có thể đọc và hiểu ngay lập tức máy đang đoán cái gì mà không cần mở tài liệu ra tra cứu ID 468 là gì.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Cảnh giao thông (traffic) chắc chắn có nhiều chủ thể (có thể có cả xe taxi, xe cảnh sát, xe buýt nhỏ - như model đang phân vân ở các rank dưới). Vì Classification chỉ xuất ra 1 nhãn, guideline (hướng dẫn gán nhãn) phải quy định rõ cách chọn chủ thể ưu tiên.
- Vì sao model score không phải ground truth? Con số score: 0.510915 (51.1%) chỉ là mức độ tự tin của máy tính (dự đoán bằng toán học) rằng ảnh này là xe taxi. Đây hoàn toàn là phỏng đoán của model.
Ground truth (nhãn thực tế) phải là sự thật khách quan do con người xác nhận dựa trên guideline (ví dụ: con người xem ảnh và tick chọn "cab"). Model có thể đoán nhầm, nên không bao giờ được lấy điểm tự tin của model làm đáp án gốc.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): class_name: "person" (người); score: 0.912625 (91.26%); bbox_xyxy: [385.33, 69.24, 498.92, 348.92]; bbox_width: 113.58 pixel; bbox_height: 279.68 pixel
- Diễn giải vị trí box bằng lời: 
1. x_min = 385.33, y_min = 69.24: Điểm bắt đầu của góc trên bên trái box, nằm sát đỉnh đầu và vai trái của người đầu bếp.
2. x_max = 498.92, y_max = 348.92: Điểm kết thúc của góc dưới bên phải box, kéo dài dọc xuống sát mép gót chân và bao trọn thân hình người này.
- So sánh số prediction ở hai threshold: +  Ở threshold thấp (ví dụ 0.35 như trong dữ liệu): Mô hình sẽ xuất ra nhiều prediction hơn. Nó bắt được cả những vật thể mà nó rất chắc chắn (người: 0.91) lẫn những vật thể nhỏ, mờ, bị che lấp mà nó ít chắc chắn hơn (các cái bát - bowl, cốc - cup với score khoảng 0.38 - 0.45).

+ Ở threshold cao (ví dụ 0.70): Số lượng prediction sẽ giảm đi đáng kể. Các box rác (false positive) hoặc các vật thể không rõ ràng sẽ bị loại bỏ, chỉ giữ lại những đối tượng to, rõ mà mô hình có độ tự tin cao (như person 0.91, bowl 0.70).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Độ bao phủ (Coverage/Recall): Khi giảm threshold, độ bao phủ tăng lên vì mô hình rà quét và không bỏ sót các vật thể nhỏ (ít bị Miss). Ngược lại, tăng threshold sẽ làm giảm độ bao phủ, dễ dẫn đến việc bỏ sót vật thể thật (False Negative).

Khối lượng công việc của Reviewer: Ngưỡng threshold thấp đồng nghĩa với việc có rất nhiều box (bao gồm cả box vẽ sai, nhận diện nhầm). Reviewer/Annotator sẽ phải tốn nhiều thời gian hơn để xóa box sai hoặc chỉnh sửa box lệch. Ngưỡng cao giúp Reviewer nhàn hơn nhưng lại có rủi ro phải tự vẽ tay bù vào những vật thể bị mô hình bỏ sót.
- Đề xuất một quy tắc box chặt:  Bounding box phải bao quanh một cách khít nhất (tightest fit) toàn bộ các phần nhìn thấy được (visible pixels) của vật thể. Bốn cạnh của box phải tiếp xúc trực tiếp với các điểm ngoài cùng (trái, phải, trên, dưới) của vật thể đó, đảm bảo không cắt lẹm vào chi tiết vật thể (thông thường cho phép chênh lệch không quá 2-3 pixel), và hạn chế tối đa việc gom không gian nền (background) vào bên trong box.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Khi vật thể không hiển thị trọn vẹn, guideline của dự án cần có quy định hoặc cơ chế escalation để giải quyết các vấn đề sau:

+ Vẽ box như thế nào? Vẽ box chỉ ôm phần đang nhìn thấy (Visible Bounding Box) hay vẽ ước lượng cả phần bị che khuất (Amodal Bounding Box)?

+ Tỷ lệ che khuất: Nếu vật thể bị che khuất hoặc cắt lẹm khỏi khung hình quá nhiều (ví dụ >70%), thì có tiếp tục gán nhãn hay bỏ qua?

+ Vật thể bị cắt làm đôi: Nếu một chiếc bát bị một vật khác chắn ngang ở giữa (chia làm 2 mảnh), guideline cần quy định là vẽ 2 box nhỏ lẻ cho 2 mảnh hay 1 box to bao trùm tất cả.

+ Escalation: Bất cứ khi nào vật thể bị che đến mức con người cũng không thể chắc chắn 100% đó là vật gì (không đủ đặc trưng nhận diện), Reviewer cần escalate lên QA hoặc Quản lý dự án để quyết định loại bỏ, tránh việc đưa dữ liệu mập mờ vào huấn luyện mô hình.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): instance_id: kitchen-001
class_name: person
score: 0.899318
bbox_xyxy: [385.45, 66.44, 498.02, 348.58]
polygon_point_count: 348 điểm
polygon_xy: danh sách các tọa độ (x, y) tạo thành đường bao của mask. Ví dụ một phần polygon bắt đầu bằng các điểm (446,70), ....
- Polygon bổ sung chi tiết gì so với box? Bounding box chỉ cho biết hình chữ nhật nhỏ nhất bao quanh đối tượng → dễ xác định vị trí nhưng chứa cả phần background.
Polygon/mask mô tả đường biên thực tế của từng đối tượng, nên thể hiện được hình dạng không đều, phần thừa của box và các vùng bị khuyết/che khuất.
- `instance_id` dùng để làm gì và không phải loại ID nào? instance_id dùng để phân biệt từng object cụ thể trong cùng một ảnh, kể cả khi chúng có cùng class_name.

Ví dụ trong kitchen có nhiều bowl, nhưng chúng được gán các ID khác nhau như kitchen-002, kitchen-003, kitchen-010; mỗi ID tương ứng với một instance riêng.
- Đề xuất một quy tắc biên mask: instance_id dùng để phân biệt từng object cụ thể trong cùng một ảnh, kể cả khi chúng có cùng class_name. Ví dụ trong kitchen có nhiều bowl, nhưng chúng được gán các ID khác nhau như kitchen-002, kitchen-003, kitchen-010; mỗi ID tương ứng với một instance riêng.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Cần có guideline rõ ràng cho các trường hợp:

biên vật thể bị mờ hoặc thiếu nét;
hai vật thể tiếp xúc hoặc chồng lên nhau;
một phần object bị che khuất;
không xác định chắc chắn phần nào thuộc object;
vật thể quá nhỏ khiến việc đặt polygon chính xác khó khăn.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

4.1. Phân loại ảnh
Đơn vị/định dạng ground truth: 1 nhãn (class) cho toàn bộ ảnh.
Lỗi hoặc điểm mơ hồ quan sát được: Cần xác định đúng nhãn của toàn ảnh theo taxonomy; không nên chỉ dựa vào một vật thể nổi bật trong ảnh.
Annotator làm gì: Đối chiếu ảnh với guideline và gán nhãn class phù hợp.
Reviewer xem gì: Kiểm tra nhãn có đúng taxonomy, đúng nội dung ảnh và có nhất quán với guideline hay không.
4.2. Phát hiện vật thể
Đơn vị/định dạng ground truth: Bounding box xyxy + class cho từng object.
Lỗi hoặc điểm mơ hồ quan sát được: Sample kitchen có nhiều vật thể, trong đó có các vật thể nhỏ và một số vật thể nằm gần hoặc chồng lấn nhau, nên có nguy cơ bỏ sót, gộp nhầm hoặc tách nhầm object.
Annotator làm gì: Vẽ một bounding box riêng cho từng instance và gán đúng class cho từng object.
Reviewer xem gì: Kiểm tra số lượng object, vị trí/kích thước bounding box, class và các object bị bỏ sót hoặc bị gộp nhầm.
4.3. Instance segmentation
Đơn vị/định dạng ground truth: class + bounding box + polygon/mask cho từng instance. Trong sample kitchen, mỗi instance có instance_id, class_name, score, bbox_xyxy và polygon_xy.
Lỗi hoặc điểm mơ hồ quan sát được: Có nhiều instance cùng class như các bowl; ngoài ra có các vật thể nhỏ như spoon, vật thể sát mép ảnh hoặc các vùng tiếp xúc/che khuất. Ví dụ kitchen-010 là một bowl có polygon chỉ gồm 12 điểm.
Annotator làm gì: Vẽ polygon bám theo biên nhìn thấy của từng object, tách riêng từng instance và không tự suy đoán phần bị che khuất khi không có đủ thông tin.
Reviewer xem gì: Kiểm tra polygon có bám sát biên object không, có ăn sang background hoặc object khác không, có thiếu/tách sai instance không; các trường hợp biên mờ, tiếp xúc hoặc che khuất không rõ thì đối chiếu guideline và yêu cầu escalation nếu cần.

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng dữ liệu/ảnh đúng phạm vi của bài toán và không chia sẻ, sao chép hoặc đưa dữ liệu nội bộ ra ngoài phạm vi được phép.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi: dừng xử lý và báo cho Reviewer/QA hoặc người phụ trách dự án để xác nhận trước khi tiếp tục.

## 6. Danh sách bằng chứng

- [x ] `classification_predictions.json`
- [ x] `detection_predictions.json`
- [x ] `segmentation_predictions.json`
- [x ] `IMAGE_ATTRIBUTION.md`
- [x ] `visuals/classification_top5.png`
- [x ] `visuals/detection_predictions.png`
- [ x] `visuals/segmentation_prediction.png`
- [x ] Ô validation cuối notebook báo `PASS`.
- [x ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
