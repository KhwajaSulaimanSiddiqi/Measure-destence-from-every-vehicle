import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import time
import math
from collections import defaultdict

# ================= 1. پرسپکتیو =================
SOURCE_POINTS = np.array([
    [923, 110],
    [970, 68],
    [1040, 139],
    [993, 184]
], dtype=np.float32)

REAL_WIDTH_M = 3.0
REAL_HEIGHT_M = 6.0

DESTINATION_POINTS = np.array([
    [0, 0],
    [REAL_WIDTH_M, 0],
    [REAL_WIDTH_M, REAL_HEIGHT_M],
    [0, REAL_HEIGHT_M]
], dtype=np.float32)

MATRIX = cv2.getPerspectiveTransform(SOURCE_POINTS, DESTINATION_POINTS)

# ================= 2. بارگذاری کلاس‌ها از فایل کوکو (با شناسه‌های درست) =================
COCO_PATH = "coco.txt"  # فایل حاوی نام کلاس‌ها به ترتیب شناسه‌های کوکو (هر خط یک نام)
# شناسه‌های کلاس‌های مورد نظر (بر اساس کوکو: 2=car, 3=motorcycle, 5=bus, 7=truck)
TARGET_CLASS_IDS = [0, 1, 2, 3]

# دیکشنری نهایی: key=class_id, value=(name, color)
VEHICLE_CLASSES = {}

try:
    with open(COCO_PATH, 'r') as f:
        # خواندن تمام خطوط (نام کلاس‌ها) به ترتیب
        class_names = [line.strip() for line in f if line.strip()]
    # برای هر شناسه مورد نظر، اگر در محدوده لیست بود، نام آن را بگیریم
    for cid in TARGET_CLASS_IDS:
        if cid < len(class_names):
            name = class_names[cid]
            # تولید رنگ یکتا و پایدار برای این کلاس
            np.random.seed(cid)
            color = tuple(int(x) for x in np.random.randint(0, 255, 3))
            VEHICLE_CLASSES[cid] = (name, color)
        else:
            print(f"⚠️ شناسه {cid} خارج از محدوده فایل {COCO_PATH} است.")
    if VEHICLE_CLASSES:
        print(f"✅ {len(VEHICLE_CLASSES)} کلاس از {COCO_PATH} بارگذاری شد: {VEHICLE_CLASSES}")
    else:
        raise FileNotFoundError
except FileNotFoundError:
    print(f"⚠️ فایل {COCO_PATH} یافت نشد یا معتبر نیست. از کلاس‌های پیش‌فرض استفاده می‌شود.")
    # مقدار پیش‌فرض (با شناسه‌های درست)
    VEHICLE_CLASSES = {
        2: ("car", (0, 255, 255)),
        3: ("motorcycle", (0, 165, 255)),
        5: ("bus", (0, 255, 0)),
        7: ("truck", (0, 0, 255)),
    }

# ================= 3. توابع =================
def get_real_point(x, y):
    pt = np.array([[[x, y]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, MATRIX)
    return dst[0][0][0], dst[0][0][1]

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_safe_color(dist):
    if dist > 10:
        return (0, 255, 0)   # سبز
    elif dist > 5:
        return (0, 165, 255) # نارنجی
    else:
        return (0, 0, 255)   # قرمز

# ================= 4. رسم نشانگر =================
def draw_marker(img, center, class_name, speed, color):
    cx, cy = center
    line_len1 = 20
    line_len2 = 20

    p2 = (cx - line_len1, cy - line_len1)
    p3 = (p2[0] - line_len2, p2[1])

    cv2.line(img, (cx, cy), p2, color, 2)
    cv2.line(img, p2, p3, color, 2)
    cv2.circle(img, (cx, cy), 6, color, -1)

    text = f"{class_name} {speed:.1f}km/h"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_x = p3[0] - 10
    text_y = p3[1] - 10
    cv2.rectangle(img,
                  (text_x - 5, text_y - th - 5),
                  (text_x + tw + 5, text_y + 5),
                  color, -1)
    cv2.putText(img, text, (text_x, text_y),
                font, font_scale, (0, 0, 0), thickness)

# ================= 5. مدل =================
model = YOLO("traffic_analysis.pt")  # مطمئن شوید فایل مدل در مسیر صحیح وجود دارد
tracker = sv.ByteTrack()
cap = cv2.VideoCapture("16.mp4")  # مسیر ویدئوی ورودی

if not cap.isOpened():
    print("❌ خطا: ویدئو باز نشد. مسیر را بررسی کنید.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (width, height))

# ================= 6. حافظه حرکت =================
history = defaultdict(list)
prev_time = time.time()

# ================= 7. پردازش =================
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    # تشخیص با مدل: فقط کلاس‌های مورد نظر را می‌گیریم
    results = model(frame, classes=TARGET_CLASS_IDS, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # فیلتر نهایی: فقط کلاس‌هایی که در VEHICLE_CLASSES تعریف شده‌اند نگه داریم
    vehicle_class_ids = list(VEHICLE_CLASSES.keys())
    detections = detections[np.isin(detections.class_id, vehicle_class_ids)]

    # بروزرسانی ردیابی
    detections = tracker.update_with_detections(detections)

    positions = {}

    if len(detections) > 0:
        for bbox, track_id, class_id in zip(detections.xyxy, detections.tracker_id, detections.class_id):
            if track_id is None:
                continue

            class_name, color = VEHICLE_CLASSES.get(class_id, ("unknown", (255,255,255)))

            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            real = get_real_point(cx, cy)

            # تاریخچه
            history[track_id].append((real, current_time))
            if len(history[track_id]) > 10:
                history[track_id].pop(0)

            # سرعت
            speed = 0
            vx, vy = 0, 0

            if len(history[track_id]) >= 2:
                (p_old, t_old) = history[track_id][-2]
                dist_move = distance(p_old, real)
                dt_move = current_time - t_old

                if dt_move > 0:
                    speed_mps = dist_move / dt_move
                    speed = speed_mps * 3.6
                    vx = (real[0] - p_old[0]) / dt_move
                    vy = (real[1] - p_old[1]) / dt_move

            positions[int(track_id)] = {
                "pixel": (cx, cy),
                "real": real,
                "velocity": (vx, vy),
                "speed": speed
            }

            draw_marker(frame, (cx, cy), class_name, speed, color)

    # تحلیل فاصله و تصادف (با محدودیت ۲۰ متر)
    ids = list(positions.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            id1, id2 = ids[i], ids[j]
            p1 = positions[id1]
            p2 = positions[id2]

            pt1 = p1["pixel"]
            pt2 = p2["pixel"]
            dist = distance(p1["real"], p2["real"])

            # فقط اگر فاصله ≤ ۲۰ متر باشد خط و متن رسم شود
            if dist <= 30.0:
                color = get_safe_color(dist)

                v1 = np.array(p1["velocity"])
                v2 = np.array(p2["velocity"])
                relative_v = v1 - v2
                relative_p = np.array(p1["real"]) - np.array(p2["real"])

                ttc = None
                if np.dot(relative_v, relative_v) > 0:
                    ttc = - np.dot(relative_p, relative_v) / np.dot(relative_v, relative_v)

                collision_risk = False
                if ttc is not None and 0 < ttc < 3 and dist < 8:
                    collision_risk = True
                    color = (0, 0, 255)

                thickness = 2 if collision_risk else 1
                cv2.line(frame, pt1, pt2, color, thickness)

                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                text = f"{dist:.1f}m"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                tx = mid_x - tw // 2
                ty = mid_y + th // 2
                overlay = frame.copy()
                cv2.rectangle(overlay, (tx-2, ty-th-2), (tx+tw+2, ty+2), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, text, (tx, ty), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("AI Safety System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()