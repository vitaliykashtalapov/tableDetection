import argparse
import time
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
# Фикс для отсутствия поддержки cv2.selectROI в Wayland
from roi_patch import select_roi

# Состояния столика
STATE_EMPTY   = "empty"    # Стол пустой — людей нет
STATE_OCCUPIED = "occupied" # Стол занят - человек в зоне
# За подход к столу считаем переход с empty в occupied


# Класс детекции людей
class PersonDetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")


    def detect(self, frame: np.ndarray, roi: tuple) -> list:
        
        # Возвращает список прямоугольников людей, центр которых попадает внутрь координат roi
       
        rx, ry, rw, rh = roi

        return self._detect_yolo(frame, rx, ry, rw, rh)

    def _detect_yolo(self, frame, rx, ry, rw, rh):
        # Класс 0 в COCO = person
        results = self.model(frame, classes=[0], verbose=False)[0]
        persons_in_roi = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # Проверяем, что центр ounding box попадает в зону столика
            if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                persons_in_roi.append((x1, y1, x2, y2))
        return persons_in_roi


# ТРЕКЕР СОСТОЯНИЙ СТОЛИКА

# Отслеживает состояние одного столика и записывает события в DataFrame
class TableStateTracker:

    N_CONFIRM = 10  # кадров для подтверждения смены состояния
    # Состояние меняется только после N_CONFIRM кадров подряд с одинаковым результатом детекции

    def __init__(self, fps: float):
        self.fps    = fps
        self.state  = STATE_EMPTY
        self.events = []

        self._pending_state = None
        self._pending_count = 0
        self._last_empty_ts : float | None = None  # timestamp последнего события "стол пуст"

    # Вызывается для каждого кадра
    def update(self, frame_idx: int, persons_in_roi: list):
        
        timestamp_sec = frame_idx / self.fps
        new_candidate = STATE_OCCUPIED if persons_in_roi else STATE_EMPTY
        # persons_in_roi - список боксов людей в зоне столика из _detect_yolo

        # Буфер подтверждения
        if new_candidate == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = new_candidate
            self._pending_count  = 1

        if self._pending_count < self.N_CONFIRM:
            return  # Ещё не достигли порога - состояние не меняем

        if new_candidate == self.state:
            return  # Состояние уже актуально

        # Фиксируем событие
        prev_state  = self.state
        self.state  = new_candidate

        event = {
            "frame"      : frame_idx,
            "timestamp"  : round(timestamp_sec, 2),
            "event"      : None,
            "delay_after_empty_sec": None,
        }

        if self.state == STATE_EMPTY:
            event["event"] = "table_empty"
            self._last_empty_ts = timestamp_sec

        elif self.state == STATE_OCCUPIED:
            if prev_state == STATE_EMPTY:
                event["event"] = "approach_after_empty"
                if self._last_empty_ts is not None:
                    event["delay_after_empty_sec"] = round(
                        timestamp_sec - self._last_empty_ts, 2
                    )
            else:
                event["event"] = "table_occupied"

        self.events.append(event)
        print(f"[EVENT] t={timestamp_sec:.1f}s | frame={frame_idx} | {event['event']}"
              + (f" | delay={event['delay_after_empty_sec']}s" if event['delay_after_empty_sec'] else ""))

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

    def mean_delay(self) -> float | None:
        df = self.get_dataframe()
        if df.empty or "event" not in df.columns:
            return None
        delays = df[df["event"] == "approach_after_empty"]["delay_after_empty_sec"].dropna()
        return float(delays.mean()) if len(delays) > 0 else None


# ВИЗУАЛИЗАЦИЯ

# Цвета (BGR)
COLOR_EMPTY    = (0,   200,  0)   # зелёный  - стол пустой
COLOR_OCCUPIED = (0,    0,  220)  # красный  - стол занят
COLOR_PERSON   = (0,  200, 255)   # жёлтый   - боксы людей

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_overlay(frame: np.ndarray,
                 roi: tuple,
                 state: str,
                 persons: list,
                 timestamp_sec: float,
                 mean_delay: float | None) -> np.ndarray:
    
    # Рисуем зону столика, боксы людей и HUD с текущим состоянием.
   
    rx, ry, rw, rh = roi
    color = COLOR_EMPTY if state == STATE_EMPTY else COLOR_OCCUPIED
    label = "EMPTY" if state == STATE_EMPTY else "OCCUPIED"

    # Зона столика
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 3)
    cv2.putText(frame, f"TABLE: {label}",
                (rx, ry - 8), FONT, 0.65, color, 2, cv2.LINE_AA)

    # Боксы людей в зоне
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)

    # HUD (верхний левый угол)
    hud_lines = [
        f"Time: {timestamp_sec:.1f}s",
        f"State: {label}",
    ]
    if mean_delay is not None:
        hud_lines.append(f"Avg delay: {mean_delay:.1f}s")

    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line, (10, 28 + i * 26),
                    FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, 28 + i * 26),
                    FONT, 0.65, (20, 20, 20), 1, cv2.LINE_AA)

    return frame


# ПАЙПЛАЙН
def run(video_path: str,
        output_path: str = "output.mp4",
        report_path: str = "report.txt",
        use_yolo: bool = True):

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"VIDEO {width}x{height} @ {fps:.1f}fps | ~{total} кадров")

    # Первый кадр для выбора ROI
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Не удалось прочитать первый кадр видео.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # перематываем в начало

    # Выбор зоны столика через патч
    roi = select_roi(first_frame.copy())

    # Инициализация классов детектора и трекера
    detector = PersonDetector()
    tracker  = TableStateTracker(fps=fps)

    # Видеозапись результата
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx   = 0
    start_wall  = time.time()

    print("\n[PROCESS] Обработка видео в режиме реального времени... (Q для остановки)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция людей в зоне столика
        persons = detector.detect(frame, roi)

        # Обновление состояния
        tracker.update(frame_idx, persons)

        # Рисуем визуализацию
        ts = frame_idx / fps
        out_frame = draw_overlay(
            frame.copy(), roi,
            tracker.state, persons, ts,
            tracker.mean_delay()
        )

        writer.write(out_frame)

        # Показываем в окне (опционально)
        cv2.imshow("Table Detection", out_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_wall
            print(f"  ... обработано {frame_idx}/{total} кадров ({elapsed:.0f}s)")

    # Завершение
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Видео сохранено: {output_path}")

    # ── Аналитика ─────────────────────────────────
    df = tracker.get_dataframe()
    mean_d = tracker.mean_delay()

    print("\n══════════════ СОБЫТИЯ ══════════════")
    print(df.to_string(index=False))
    print("═════════════════════════════════════")

    if mean_d is not None:
        print(f"\n✅ Среднее время между уходом гостя и подходом следующего: {mean_d:.2f} сек\n")
    else:
        print("\n!Недостаточно данных для вычисления средней задержки!\n")

    # Сохраняем CSV событий
    csv_path = Path(output_path).stem + "_events.csv"
    df.to_csv(csv_path, index=False)

    # Текстовый отчёт
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Отчёт системы детекции подхода к столикам ===\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Видео: {video_path}\n")
        f.write(f"Зона ROI (x,y,w,h): {roi}\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"Обработано кадров: {frame_idx}\n\n")
        f.write("--- Событийный лог ---\n")
        f.write(df.to_string(index=False))
        f.write("\n\n--- Статистика ---\n")
        if mean_d is not None:
            f.write(f"Среднее время задержки (пусто - подход): {mean_d:.2f} сек\n")
            delays = df[df["event"] == "approach_after_empty"]["delay_after_empty_sec"].dropna()
            f.write(f"Мин. задержка: {delays.min():.2f} сек\n")
            f.write(f"Макс. задержка: {delays.max():.2f} сек\n")
            f.write(f"Количество подходов: {len(delays)}\n")
        else:
            f.write("Недостаточно данных.\n")

    return df, mean_d


# Точка входа
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Детекция подхода к столикам по видео")
    parser.add_argument("--video",   required=True,          help="Путь к входному видео")
    parser.add_argument("--output",  default="output.mp4",   help="Путь к выходному видео")
    parser.add_argument("--report",  default="report.txt",   help="Путь к текстовому отчёту")
    args = parser.parse_args()

    run(
        video_path  = args.video,
        output_path = args.output,
        report_path = args.report,
    )
