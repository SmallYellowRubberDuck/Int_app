import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue


# Функция для обработки модели сегментации дорожных разметок с ONNX
def Run_onnx_lane(model, img):
    img = cv2.resize(img, (320, 320))
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    output = model.run(None, {"input": img})
    x0, x1 = output[0], output[1]

    da_predict = np.argmax(x0, axis=1)[0] * 255
    ll_predict = np.argmax(x1, axis=1)[0] * 255
    lane_img = np.zeros_like(img[0].transpose(1, 2, 0))
    lane_img[da_predict > 100] = [255, 0, 0]
    lane_img[ll_predict > 100] = [0, 255, 0]
    return lane_img[:, :, 1]

def segment_lane_lines(lane_model, frame):
    return Run_onnx_lane(lane_model, frame)

def yolo_predict(model, frame, conf_threshold=0.47):
    # Подготовка кадра
    resized_frame = cv2.resize(frame, (320, 320)).astype(np.float32)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    resized_frame = np.transpose(resized_frame, (2, 0, 1)) / 255.0
    input_frame = np.expand_dims(resized_frame, axis=0)

    # Запуск модели ONNX и получение предсказаний
    outputs = model.run(None, {'images': input_frame})
    predictions = outputs[0] 

    boxes = []
    
    for i in range(predictions.shape[2]):
        pred = predictions[0, :, i]
        box = pred[:4]  # x_center, y_center, width, height
        scores = pred[4:]  # вероятности классов

        # Находим индекс и значение максимальной вероятности класса
        prob = scores.max()
        class_id = scores[4:].argmax()

        # Применяем порог вероятности
        if prob > conf_threshold:
            x_center, y_center, width, height = box
            # Преобразование координат в x1, y1, x2, y2
            x1 = int((x_center - width / 2) / 320 * frame.shape[1])
            y1 = int((y_center - height / 2) / 320 * frame.shape[0])
            x2 = int((x_center + width / 2) / 320 * frame.shape[1])
            y2 = int((y_center + height / 2) / 320 * frame.shape[0])

            boxes.append([x1, y1, x2, y2, class_id])

    return np.array(boxes)  # Возвращаем массив боксов

# Функция чтения кадров с видео и добавления их в очередь
def read_frames(video_path, frame_queue):
    video = cv2.VideoCapture(video_path)
    last_processed_second = -1

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Определяем текущую секунду кадра
        current_time_in_seconds = int(video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        
        # Добавляем только один кадр на каждую секунду
        if current_time_in_seconds != last_processed_second:
            last_processed_second = current_time_in_seconds
            frame_queue.put((current_time_in_seconds, frame))
    
    frame_queue.put((None, None))  # Флаг завершения
    video.release()

def process_frame(frame, sign_model, traffic_lights_model, lane_model):
    resized_frame = cv2.resize(frame, (320, 320))

    # Преобразуем изображение в оттенки серого и усиливаем контраст
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = {
            "signs": executor.submit(yolo_predict, sign_model, resized_frame),
            "traffic_lights": executor.submit(yolo_predict, traffic_lights_model, resized_frame),
            "lane_lines": executor.submit(segment_lane_lines, lane_model, resized_frame)
        }
        
    sign_layer, traffic_light_layer, lane_line_layer = np.zeros((320, 320)), np.zeros((320, 320)), results["lane_lines"].result()

    for box in results["signs"].result():
        x1, y1, x2, y2, class_id = map(int, box[:5])
        if class_id in {7, 14, 20, 17}:
            sign_layer[y1:y2, x1:x2] = class_id + 1

    for box in results["traffic_lights"].result():
        x1, y1, x2, y2, class_id = map(int, box[:5])
        traffic_light_layer[y1:y2, x1:x2] = class_id + 1

    # Склеиваем слои, добавляя чёрно-белый слой с контрастом
    combined_layer = sign_layer + traffic_light_layer

    return np.stack([combined_layer, lane_line_layer, gray_frame], axis=-1)


def process_video(video_path, sign_model, traffic_lights_model, lane_model, num_workers=4):
    frame_queue = Queue(maxsize=15)
    processed_frames = []

    threading.Thread(target=read_frames, args=(video_path, frame_queue)).start()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_frame, frame, sign_model, traffic_lights_model, lane_model): sec for sec, frame in iter(lambda: frame_queue.get(), (None, None))}
        
        for future in as_completed(futures):
            second = futures[future]
            processed_frames.append((second, future.result()))

    processed_frames.sort(key=lambda x: x[0])
    return [frame for _, frame in processed_frames]
