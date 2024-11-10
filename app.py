import math
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox, ttk
import pandas as pd
import os
import cv2
import platform
import psutil
import torch

def ml_model(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров
    count_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Общее количество кадров
    duration = count_frame // fps  # Длительность видео в секундах
    return {
        "Video": os.path.basename(video_path),
        "Result": "Processed",
        "Duration (s)": duration,
        "Count Frame": count_frame
    }

class VideoProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processing App")
        self.root.geometry("1280x800")
        self.root.configure(bg="#F4F6F9")
        
        # Центрируем главное окно
        self.center_window(self.root)
        self.model = "Model1"
        self.video_files = []

        # Меню
        self.menu = tk.Menu(root)
        root.config(menu=self.menu)

        # Подменю для выбора конфигурации ПК
        self.pc_config_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="PC Configuration", menu=self.pc_config_menu)
        self.choose_model_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Model", menu=self.choose_model_menu)
        # Кнопка для дообучения модели
        self.further_train_button = tk.Button(root, text="Train Model", command=self.load_files_window, relief="raised", width=30, height=2, bg="#A9C8FF", fg="white", font=("Arial", 12, "bold"), activebackground="#FB8C00", activeforeground="white")
        self.further_train_button.pack(pady=15)
        
        # Кнопка для загрузки видео
        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video, relief="raised", width=20, height=2, bg="#8320F5", fg="white", font=("Arial", 12, "bold"), activebackground="#6C94DC", activeforeground="white")
        self.upload_button.pack(pady=15)

        # Кнопка для обработки видео
        self.process_button = tk.Button(root, text="Process Video", command=self.process_video, relief="raised", width=20, height=2, bg="#6C94DC", fg="white", font=("Arial", 12, "bold"), activebackground="#A9C8FF", activeforeground="white")
        self.process_button.pack(pady=15)

        # Кнопка для сохранения результатов
        self.save_button = tk.Button(root, text="Save Results", command=self.save_results, relief="raised", width=20, height=2, bg="#A9C8FF", fg="white", font=("Arial", 12, "bold"), activebackground="#8320F5", activeforeground="white")
        self.save_button.pack(pady=15)

        # Таблица результатов
        self.results_table = None

        # Treeview для отображения результатов
        self.tree = ttk.Treeview(root, columns=("Video", "Result", "Duration (s)", "Count Frame"), show="headings", height=10)
        self.tree.heading("Video", text="Video", anchor="w")
        self.tree.heading("Result", text="Result", anchor="w")
        self.tree.heading("Duration (s)", text="Duration (s)", anchor="w")
        self.tree.heading("Count Frame", text="Count Frame", anchor="w")

        # Устанавливаем стиль для Treeview
        self.tree.tag_configure("evenrow", background="#F1F1F1")
        self.tree.tag_configure("oddrow", background="#FFFFFF")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=10)

        # Конфигурации по умолчанию
        self.get_processor_info()
        self.get_memory_info()
        self.check_cuda_availability()

        # Добавляем выпадающие списки для выбора конфигурации ПК
        self.processor_menu_item = self.pc_config_menu.add_command(label=f"Processor: {self.processor.get()}", command=self.select_processor)
        self.ram_menu_item = self.pc_config_menu.add_command(label=f"RAM: {self.ram.get()}", command=self.select_ram)
        self.gpu_menu_item = self.pc_config_menu.add_command(label=f"Graphics Card: {self.gpu.get()}", command=self.select_gpu)

        # Добавляем выбор моделей в соответствующее меню 
        self.model1_menu_item = self.choose_model_menu.add_command(label="Model1", command=lambda: self.choose_model('Model1'))
        self.model2_menu_item = self.choose_model_menu.add_command(label="Model2", command=lambda: self.choose_model('Model2'))
        #...self.model3_menu_item = self.choose_model_menu.add_command(label="Model3", command=self.choose_model('Model3'))
    def choose_model(self, model):
        print(self.model, model)
        self.model = model

    # Открытие окна выбора данных для дообучения модели
    def load_files_window(self):
            top = tk.Toplevel(self.root)
            top.title("Choose video folder and markup excel file")
            top.grab_set()
            # Метки и кнопки для выбора видео папки и xlsx файла
            video_folder_label = tk.Label(top, text="Select Video Folder:", font=("Arial", 12))
            video_folder_label.pack(padx=10, pady=5)

            self.video_folder_path = tk.StringVar()
            video_folder_button = tk.Button(top, text="Browse", command=self.select_video_folder, width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
            video_folder_button.pack(pady=5)

            self.xlsx_file_path = tk.StringVar()
            xlsx_label = tk.Label(top, text="Select Excel File:", font=("Arial", 12))
            xlsx_label.pack(padx=10, pady=5)
            
            xlsx_button = tk.Button(top, text="Browse", command=self.select_xlsx_file, width=20, height=2, bg="#2196F3", fg="white", font=("Arial", 12, "bold"))
            xlsx_button.pack(pady=5)
            #Создадим пробел для разграничения ввода-вывода
            empty_space = tk.Label(top, text="", height=1)
            empty_space.pack(pady=10)
            # Кнопка для подтверждения окончания выбора
            close_button = tk.Button(top, text="Confirm", command=lambda: self.close_train_window(top), width=20, height=2, bg="#FF9800", fg="white", font=("Arial", 12, "bold"))
            close_button.pack(pady=10)
            # Кнопка отмены
            cancel_button = tk.Button(top, text="Cancel", command = top.destroy, width=20, height=2, bg="#FF5722", fg='white', font=("Arial", 12, "bold"))
            cancel_button.pack(pady=10)
            self.center_window(top)
    # Функция получения от пользователя пути к папке с видео для дообучения
    def select_video_folder(self):
        folder_path = filedialog.askdirectory(title="Select Video Folder")
        if folder_path:
            self.video_folder_path.set(folder_path)
    # Функция получения от пользователя пути к файлу разметки для дообучения       
    def select_xlsx_file(self):
        file_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.xlsx_file_path.set(file_path)
    # Функция подтверждения выбора файлов для дообучения и закрытия окна
    def close_train_window(self, top):
        if self.video_folder_path.get() and self.xlsx_file_path.get():
            messagebox.showinfo("Info", f"You selected {self.video_folder_path.get()} as video folder\n and \n{self.xlsx_file_path.get()} as markup file")
            top.grab_release()
            messagebox.showinfo("Info", "Wait until next warning with result of learning")
            # СЮДА ВСТАВИТЬ ОБУЧЕНИЕ МОДЕЛИ
            messagebox.showwarning("Warning", "Model is ready")
            top.destroy()
        else:
            messagebox.showwarning("Warning", "Please, select both video folder and Excel file")
    # Функция центрирования окна
    def center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        
        # Вычисляем позицию для центрирования окна
        position_top = int(screen_height / 2 - height / 2)
        position_left = int(screen_width / 2 - width / 2)
        
        window.geometry(f'{width}x{height}+{position_left}+{position_top}')
    # Функция получения информации о процессоре
    def get_processor_info(self):
        processor_info = platform.processor()
        if 'Intel' in processor_info:
            self.processor = tk.StringVar(value="Intel")
        elif 'AMD' in processor_info:
            self.processor = tk.StringVar(value="AMD")
        elif 'Apple' in processor_info:
            self.processor = tk.StringVar(value="Apple")
        else:
            self.processor = "Unknown"
    # Функция получения объема оперативной памяти
    def get_memory_info(self):
        mem = psutil.virtual_memory()
        self.ram = tk.StringVar(value=f"{math.ceil(mem.total / 1024 ** 3)} GB")
    # Функция проверки возможности использования CUDA-ядер
    def check_cuda_availability(self):
        if torch.cuda.is_available():
            self.gpu = tk.StringVar(value="With CUDA")
        else:
            self.gpu = tk.StringVar(value="Without CUDA")
    # Блок функций для выбора желаемых характеристик
    def select_processor(self):
        processor_options = ["AMD", "Intel", "Apple"]
        self.show_option_menu("Select Processor", self.processor, processor_options)

    def select_ram(self):
        ram_options = ["4 GB", "8 GB", "16 GB", "32 GB", "64 GB", "128 GB"]
        self.show_option_menu("Select RAM", self.ram, ram_options)

    def select_gpu(self):
        gpu_options = ["With CUDA", "Without CUDA"]
        self.show_option_menu("Select GPU type", self.gpu, gpu_options)

    def show_option_menu(self, title, variable, options):
        top = tk.Toplevel(self.root)
        top.title(title)
        
        # Центрируем всплывающее окно относительно главного окна
        self.center_window(top)
        
        combobox = ttk.Combobox(top, textvariable=variable, values=options, state="readonly", font=("Arial", 12), width=20)
        combobox.pack(padx=20, pady=20)
        combobox.set(variable.get())  # Устанавливаем текущее значение

        def on_select():
            top.destroy()
            self.pc_config_menu.delete(0, tk.END)
            self.processor_menu_item = self.pc_config_menu.add_command(label=f"Processor: {self.processor.get()}", command=self.select_processor)
            self.ram_menu_item = self.pc_config_menu.add_command(label=f"RAM: {self.ram.get()}", command=self.select_ram)
            self.gpu_menu_item = self.pc_config_menu.add_command(label=f"Graphics Card: {self.gpu.get()}", command=self.select_gpu)
            messagebox.showinfo("Selected", f"You selected: {variable.get()}")

        select_button = tk.Button(top, text="Select", command=on_select, width=15, height=2, bg="#6C94DC", fg="white", font=("Arial", 12, "bold"), activebackground="#A9C8FF", activeforeground="white")
        select_button.pack(pady=10)
    # Функция опроса пользователя на предмет видео для работы
    def upload_video(self):
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.wmv;*.avchd;*.swf;*.flv;*.mkv;*.webm;*.mpeg2")])
        self.video_files.extend(files)
        messagebox.showinfo("Info", f"Uploaded {len(files)} video(s).")
    # Функция обработки видео моделью
    def process_video(self):
        print(self.model)
        if not self.video_files:
            messagebox.showwarning("Warning", "No video files uploaded!")
            return
        results = []
        for video in self.video_files:
            result = ml_model(video)
            results.append(result)
        # Сохраняем результаты в таблице
        self.results_table = pd.DataFrame(results)
        #messagebox.showinfo("Info", "Processing complete!")
        # Очищаем Treeview перед добавлением новых данных
        for item in self.tree.get_children():
            self.tree.delete(item)
        # Добавляем новые данные в Treeview
        for index, row in self.results_table.iterrows():
            tag = "oddrow" if index % 2 == 0 else "evenrow"
            self.tree.insert("", "end", values=(row["Video"], row["Result"], row["Duration (s)"], row["Count Frame"]), tags=(tag,))
    # Функция зааписи результатов работы модели в xslx-файл
    def save_results(self):
        if self.results_table is None:
            messagebox.showwarning("Warning", "No results to save!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.results_table.to_excel(file_path, index=False)
            messagebox.showinfo("Info", f"Results saved to {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessingApp(root)
    root.mainloop()