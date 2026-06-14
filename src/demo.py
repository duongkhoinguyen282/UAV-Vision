import cv2
import time
from ultralytics import YOLO
import socket
import json

# Setup Trạm phát sóng UDP
UDP_IP = "127.0.0.1"  # Hiện tại test chung máy thì xài Localhost. Lên drone có 2 board thì đổi IP.
UDP_PORT = 8888       # Cổng phong thủy
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"📡 Đã mở trạm phát UDP tại {UDP_IP}:{UDP_PORT}")

# 1. Triệu hồi bộ não mAP 30%
model = YOLO("best.pt")

# 2. Mở file video
video_path = "drone_test.mp4"
cap = cv2.VideoCapture(video_path)

# 1. Lấy thông số gốc của video
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = int(cap.get(cv2.CAP_PROP_FPS))

# 2. Khai báo trần giới hạn (Ép về max 1080p để máy chạy cho mượt)
MAX_W = 1920
MAX_H = 1080

# 3. Tính toán linh hoạt
if orig_w > MAX_W or orig_h > MAX_H:
    # Lấy tỷ lệ bóp nhỏ nhất để không bị tràn viền   
    scale = min(MAX_W / orig_w, MAX_H / orig_h)
    target_w = int(orig_w * scale)
    target_h = int(orig_h * scale)
    print(f"⚠️ Video quá to ({orig_w}x{orig_h}). Đang tự động ép về {target_w}x{target_h} để chạy cho bốc!")
else:
    # Video vốn đã bé hơn 1080p thì giữ nguyên
    target_w = orig_w
    target_h = orig_h
    print(f"✅ Video chuẩn bài ({orig_w}x{orig_h}). Đéo cần bóp!")

# 4. Tạo file Output chuẩn form
out = cv2.VideoWriter('output_demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (target_w, target_h))

# 3. Từ điển Trick Toán Học (Giả sử Focal Length F = 800)
FOCAL_LENGTH = 800
# VisDrone classes cơ bản: 0:pedestrian, 1:people, 3:car, 4:van, 5:truck, 9:motor
def get_real_width(class_id):
    if class_id in [3, 4, 5]: # Ô tô, xe tải (to)
        return 2.0  # Rộng tầm 2 mét
    elif class_id in [0, 1]:  # Người đi bộ
        return 0.5  # Rộng tầm 0.5 mét
    else:
        return 0.8  # Xe máy, xe đạp... mặc định 0.8 mét

print("🚀 Bắt đầu render Demo...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break 

    # Tự động Resize theo kích thước an toàn vừa tính ở trên
    if orig_w > MAX_W or orig_h > MAX_H:
        frame = cv2.resize(frame, (target_w, target_h))

    # Bấm giờ đo FPS
    start_time = time.time()
    
    # 4. Chạy model với thuật toán Tracking ByteTrack tích hợp
    # persist=True để nó nhớ ID giữa các frame
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    # 5. Xử lý data
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        # Tạo 1  list rỗng để chứa data của toàn bộ vật thể trong 1 frame
        frame_payload = []

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Tính Center X, Y (Tâm của vật thể)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            w_pixel = x2 - x1
            w_real = get_real_width(class_id)
            distance_z = (w_real * FOCAL_LENGTH) / w_pixel if w_pixel > 0 else 0
            
	    # 1. LOGIC PHÂN LOẠI NGUY HIỂM (Giả sử < 15m là sắp đâm)
            if distance_z < 15.0:
                box_color = (0, 0, 255)       # Đỏ chóe (Báo động)
                text_color = (0, 0, 255)      
                status = "DANGER_COLLISION"
                # In chữ WARNING to chà bá giữa màn hình
                cv2.putText(frame, "!!! COLLISION WARNING !!!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            elif distance_z < 30.0:
                box_color = (0, 165, 255)     # Cam (Chú ý)
                text_color = (0, 165, 255)
                status = "WARNING"
            else:
                box_color = (0, 255, 0)       # Xanh lá (An toàn)
                text_color = (0, 255, 255)
                status = "SAFE"

            # 2. VẼ VỜI LÀM MÀU DỰA TRÊN TRẠNG THÁI
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            class_name = model.names[class_id]
            label = f"{class_name}_{track_id} | Z: {distance_z:.1f}m"
            # Đẩy cái chữ label lên xíu và dùng màu tương ứng
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # 3. TẠO CỤC JSON CHO NHÓM 3
            obj_data = {
                "id": track_id,
                "class": class_name,
                "cx": cx,
                "cy": cy,
                "z": round(distance_z, 2),
                "status": status
            }
            frame_payload.append(obj_data)
                    
        # 🚀 BẮN DATA CHO NHÓM 3
        if frame_payload:
            # Biến mảng Python thành chuỗi JSON rồi mã hóa byte
            message = json.dumps(frame_payload).encode('utf-8')
            # Khạc thẳng qua cổng mạng UDP
            sock.sendto(message, (UDP_IP, UDP_PORT))
            
            # Print nhẹ ra terminal để thấy là có chạy
            print(f"Bắn JSON: {message.decode('utf-8')} -> Nhóm 3", end='\r')
            
    # Tính FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Hiện lên màn hình và ghi vào file mp4
    # cv2.imshow("Drone Tech Lead Demo", frame)
    out.write(frame)
    print(f"Đang nhai mượt mà, ráng đợi tí...", end='\r')

    # Bấm 'q' để thoát sớm nếu thích
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
print("🎉 Render xong! Xem file output_demo.mp4 đi!")