"""
Патч для select_roi (т.к. selectROI не запускается на Wayland)

Варианты:
1. Принудительный X11/xcb  — запустить cv2.selectROI с принудительным xcb:
QT_QPA_PLATFORM=xcb python main3.py --video "NAME.mp4"
2. Matplotlib   — интерактивный выбор через plt.ginput (работает в Wayland)
3. Ручной ввод  — просто ввести x y w h в консоли (всегда работает)
"""

import os
# import sys
import cv2
import numpy as np


def select_roi_matplotlib(frame: np.ndarray) -> tuple:
    import matplotlib
    matplotlib.use("TkAgg") # или Qt5Agg
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(
        "Определяем координаты параллелограмма ROI по верхнему левому и нижнему правому углу\n"
        "(Сделать два клика и закрыть окно)",
        fontsize=12
    )
    plt.tight_layout()

    coords = []

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        coords.append((int(event.xdata), int(event.ydata)))
        ax.plot(event.xdata, event.ydata, "ro", markersize=8)
        if len(coords) == 2:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            rect = mpatches.Rectangle(
                (min(x1, x2), min(y1, y2)),
                abs(x2 - x1), abs(y2 - y1),
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(coords) < 2:
        print("Не было координат с графического ввода, предлагаем ввестии их вручную...")
        return select_roi_manual(frame)

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    print(f"Выбрана зона: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    return roi


def select_roi_manual(frame: np.ndarray) -> tuple:
    
    # Сохраняем первый кадр как PNG и просим ввести координаты
   
    preview_path = "roi_preview.png"
    cv2.imwrite(preview_path, frame)
    print(f"\nПервый кадр сохранён: {preview_path}\n")

    h, w = frame.shape[:2]
    print(f"Размер кадра: {w}x{h} пикселей")
    print("Введите координаты ROI:\n")

    try:
        x = int(input("  x (левый край, пиксели): ").strip())
        y = int(input("  y (верхний край, пиксели): ").strip())
        ww = int(input("  w (ширина зоны, пиксели): ").strip())
        hh = int(input("  h (высота зоны, пиксели): ").strip())
    except (ValueError, EOFError) as e:
        raise ValueError(f"Некорректный ввод: {e}")

    roi = (x, y, ww, hh)
    print(f"\nROI установлен: x={x}, y={y}, w={ww}, h={hh}\n")
    return roi


def select_roi(frame: np.ndarray) -> tuple:
    
    # Поочерёдно перебираем: cv2.selectROI - matplotlib - ручной ввод.

    ### cv2.selectROI
    # Для активации этого варианта запускать с переменной окружения QT_QPA_PLATFORM=xcb (пример в начале)
    backend = os.environ.get("QT_QPA_PLATFORM", "")
    try:
        print("Нажмите ENTER/SPACE для подтверждения, C — для отмены")
        roi = cv2.selectROI(
            "Выберите зону столика", frame,
            fromCenter=False, showCrosshair=True
        )
        cv2.destroyWindow("Выберите зону столика")

        if roi == (0, 0, 0, 0):
            raise RuntimeError("empty roi")

        print(f"Выбрана зона: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        return roi

    except Exception as e:
        print(f"cv2.selectROI недоступен ({e}).")

    ### matplotlib
    try:
        return select_roi_matplotlib(frame)
    except Exception as e:
        print(f"[ROI] Matplotlib недоступен ({e}).")

    ### ручной ввод
    return select_roi_manual(frame)
