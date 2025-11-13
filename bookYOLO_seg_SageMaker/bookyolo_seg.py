import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
import boto3
import concurrent.futures

MIN_W = 10
MIN_H = 10

def process_item(idx, item, image, scale_x, scale_y):

    (y_box, x_box, c_obj) = item
    rect = cv2.minAreaRect(c_obj)
    (cx, cy), (w, h), angle = rect

    # scale, warpAffine
    cx *= scale_x
    cy *= scale_y
    w *= scale_x
    h *= scale_y
    rect = ((cx, cy), (w, h), angle)

    cropped_img = crop_rotated_box_no_trim(image, rect)

    if cropped_img is None:
        return None

    _, img_encoded = cv2.imencode('.jpg', cropped_img)
    img_bytes = img_encoded.tobytes()

    # S3 업로드
    s3_key = f"crop_{idx}.jpg"

    return (rect, img_bytes, s3_key)

def model_int(model_path):
    model = YOLO(model_path)
    return model
def upload_to_s3(s3, s3_bucket_name, img_bytes, s3_key):
    try:
        s3.put_object(Bucket=s3_bucket_name, Key=s3_key, Body=img_bytes, ContentType='image/jpeg')
        print(f"Uploaded to s3://{s3_bucket_name}/{s3_key}")
    except NoCredentialsError:
        print("AWS 자격 증명을 찾을 수 없습니다.")
    except Exception as e:
        print(f"S3 업로드 중 오류 발생: {e}")
def crop_rotated_box_no_trim(orig_image, rect):
    (cx, cy), (w, h), angle = rect
    if angle < -45:
        angle += 90
        w, h = h, w

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    H, W = orig_image.shape[:2]

    corners = np.array([
        [0,    0],
        [W-1,  0],
        [W-1,  H-1],
        [0,    H-1]
    ], dtype=np.float32)

    transformed_corners = cv2.transform(corners[None,:,:], M)[0]
    min_x = transformed_corners[:,0].min()
    max_x = transformed_corners[:,0].max()
    min_y = transformed_corners[:,1].min()
    max_y = transformed_corners[:,1].max()

    needed_w = int(np.ceil(max_x - min_x))
    needed_h = int(np.ceil(max_y - min_y))

    shift_x = -min_x
    shift_y = -min_y
    M_shifted = M.copy()
    M_shifted[0,2] += shift_x
    M_shifted[1,2] += shift_y

    if needed_w < 1 or needed_h < 1:
        return None

    rotated = cv2.warpAffine(
        orig_image,
        M_shifted,
        (needed_w, needed_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0)
    )

    w_i = int(w)
    h_i = int(h)

    p = np.array([[cx],[cy],[1]], dtype=np.float32)
    new_center = M_shifted @ p
    cx_n, cy_n = new_center[0], new_center[1]

    x1 = int(cx_n - w_i/2)
    y1 = int(cy_n - h_i/2)
    x2 = x1 + w_i
    y2 = y1 + h_i

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(needed_w, x2)
    y2 = min(needed_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    cropped = rotated[y1:y2, x1:x2].copy()

    hh, ww = cropped.shape[:2]
    if ww > hh:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    return cropped

def bookyolo_seg(
    model,
    image,
    conf_thres=0.5,
    median_mode=False,
    s3_bucket_name='',
    s3_folder='',
    worker = 6
):
    s3 = boto3.client('s3')
    start_time = time.time()  # 시작 시점

    results = model.predict(
        source= image,
        save=False,
        imgsz=864,
        conf=conf_thres,
        device='0',
        show=False,
        save_crop=False,
        retina_masks=True,
        max_det=100
    )

    inf_time = time.time() - start_time  # 종료 시점 - 시작 시점
    print(f"프리딕트 시간: {inf_time:.4f}초")


    img_h, img_w = image.shape[:2]
    if not results[0].masks:
        return print('탐지 실패')

    masks_data = results[0].masks.data.cpu().numpy()
    pred_h, pred_w = masks_data.shape[1], masks_data.shape[2]
    scale_x = img_w / float(pred_w)
    scale_y = img_h / float(pred_h)

    all_contours = []
    for i in range(masks_data.shape[0]):
        mask = (masks_data[i] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            all_contours.append((c, area))

    if not all_contours:
        return print('지역화 실패')

    areas = [x[1] for x in all_contours]
    if median_mode:
        median_val = np.median(areas)
        area_threshold = median_val * 0.3
    else:
        area_threshold = np.percentile(areas, 15)

    valid_list = []
    for c, c_area in all_contours:
        if c_area < area_threshold:
            continue

        x_box, y_box, w_box, h_box = cv2.boundingRect(c)
        w_box_scaled = w_box * scale_x
        h_box_scaled = h_box * scale_y
        if w_box_scaled < MIN_W or h_box_scaled < MIN_H:
            continue

        valid_list.append((y_box, x_box, c))

    # valid_list.sort(key=lambda item: (item[0], item[1]))
    valid_list.sort(key=lambda item: item[1])

    result_list = []
    encode_results = []

    rot_time = time.time()  # 시작 시점

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        future_list = []
        for idx, item in enumerate(valid_list):
            f = executor.submit(process_item, idx, item, image, scale_x, scale_y)
            future_list.append(f)

        for fut in concurrent.futures.as_completed(future_list):
            rect, img_bytes, s3_key = fut.result()
            if rect is not None and img_bytes is not None:
                encode_results.append((rect, img_bytes, s3_key))

    rot_to_time = time.time() - rot_time  # 종료 시점 - 시작 시점
    print(f"로테이션 시간: {rot_to_time:.4f}초")

    s3_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker) as tpool:
        futures = []
        for (rect, img_bytes, s3_key) in encode_results:
            full_s3_key = f"{s3_folder}{s3_key}"
            fut = tpool.submit(upload_to_s3, s3, s3_bucket_name, img_bytes, full_s3_key)
            futures.append(fut)

        concurrent.futures.wait(futures)

        for i, fut in enumerate(futures):
            fut.result()
            result_list.append(encode_results[i][0])

    # for (rect, img_bytes, s3_key) in encode_results:
    #     full_s3_key = f"{s3_folder}{s3_key}"
    #     upload_to_s3(s3, s3_bucket_name, img_bytes, full_s3_key)
    #     result_list.append(rect)

    s3_to_time = time.time() - s3_time  # 종료 시점 - 시작 시점
    print(f"s3 시간: {s3_to_time:.4f}초")

    # for item in valid_list:
    #     (y_box, x_box, c_obj) = item
    #     rect = cv2.minAreaRect(c_obj)
    #     (cx, cy), (w, h), angle = rect
    #
    #     cx *= scale_x
    #     cy *= scale_y
    #     w  *= scale_x
    #     h  *= scale_y
    #     rect = ((cx, cy), (w, h), angle)
    #
    #     cropped_img = crop_rotated_box_no_trim(image, rect)
    #
    #     image_filename = f'crop_{n + 1}.jpg'
    #     s3_key = s3_folder + '/' + image_filename
    #
    #     _, img_encoded = cv2.imencode('.jpg', cropped_img)
    #     img_bytes = img_encoded.tobytes()
    #
    #     upload_to_s3(s3, s3_bucket_name, img_bytes, s3_key)
    #     result_list.append(rect)
    #
    #     n += 1

    total_time = time.time() - start_time  # 종료 시점 - 시작 시점
    print(f"전체 인퍼런스 시간: {total_time:.4f}초")
    print("저장 완료")

    return result_list