import os
import cv2
import pytesseract
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilenames, askdirectory
import sys
import numpy
import tempfile

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

root = Tk()
root.withdraw()

print("Выберите файлы для обработки")
files = askopenfilenames(
    title = "Выберите нужные файлы",
    filetypes = [("Images", "*.jpg *jpeg *.png *.JPG *.PNG")]
)

if not files:
    print("Нет файлов для обратоки!")
    exit()

print("Выберите папку сохранения")
output = askdirectory(
    title = "Выберите папку куда сохранять изображения"
)

if not output:
    print("Папка для сохранения не выбрана!")
    exit()

tesseract = resource_path("Tesseract-OCR")
pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract, "tesseract.exe")
tesseract_conf = "--psm 11 --oem 3 -c tessedit_char_whitelist=АВЕКМНОРСТУХ0123456789"

for file_path in files:
    file_name = os.path.basename(file_path)
    print(f"Работаю над файлом: {file_name} ...")

    image = cv2.imdecode(numpy.fromfile(file_path, dtype=numpy.uint8), cv2.IMREAD_COLOR)

    if image is None:
        print(f"Невозможно обработать {file_name}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = resource_path("haarcascade_russian_plate_number.xml")

    with open(cascade, "rb") as file:
        data = file.read()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
    tmp.write(data)
    tmp.close()

    detect = cv2.CascadeClassifier(cascade)

    plate = detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12, minSize=(30,30))

    if len(plate)>0:
        x, y, w, h = plate[0]

        roi_gray = gray[y:y+h, x:x+w]
        roi_large = cv2.resize(roi_gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, roi_thresh = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        txt = pytesseract.image_to_string(roi_thresh, config=tesseract_conf).strip()

        if txt and len(txt)>= 3:
            res = f"[{txt}]"
            roi_color = image[y:y + h, x:x + w]
            blur = cv2.GaussianBlur(roi_color, (129, 129), 0)
            image[y:y + h, x:x + w] = blur

            save = os.path.join(output, file_name)
            cv2.imencode(os.path.splitext(save)[1], image)[1].tofile(save)
            print(f"Выполнена обработка: номер {res} обработан")

        else:
            new_name = "Не_Распознано_" + file_name
            save = os.path.join(output, new_name)
            cv2.imencode(os.path.splitext(save)[1], image)[1].tofile(save)
            print(f"Не удалось распознать номер в {file_name}.\n"
                  f"Изображение было сохранено как {new_name}\n")
    else:
        new_name = "Не_Найдено_" + file_name
        save = os.path.join(output, new_name)
        cv2.imencode(os.path.splitext(save)[1], image)[1].tofile(save)
        print(f"Не удалось найти номер в {file_name}.\n"
              f"Изображение было сохранено как {new_name}\n")

print("\n Обработка выполнена!")
ans = messagebox.askyesno(
    title="Обработка выполнена!",
    message="Хотите открыть папку с файлами?"
)

if ans:
    os.startfile(os.path.normpath(output))
else:
    exit()