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

        self.video_files = []

        # Меню
        self.menu = tk.Menu(root)
        root.config(menu=self.menu)
        
        # Подменю для выбора конфигурации ПК
        self.pc_config_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="PC Configuration", menu=self.pc_config_menu)

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

    def get_memory_info(self):
        mem = psutil.virtual_memory()
        self.ram = tk.StringVar(value=f"{math.ceil(mem.total/1024**3)} GB")
    def check_cuda_availability(self):
        if torch.cuda.is_available():
            self.gpu = tk.StringVar(value="With CUDA")
        else:
            self.gpu = tk.StringVar(value="Without CUDA")

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

        select_button = tk.Button(top, text="Select", command=on_select, width=15, height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), activebackground="#45a049", activeforeground="white")
        select_button.pack(pady=10)

    def upload_video(self):
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.wmv;*.avchd;*.swf;*.flv;*.mkv;*.webm;*.mpeg2")])
        self.video_files.extend(files)
        messagebox.showinfo("Info", f"Uploaded {len(files)} video(s).")

    def process_video(self):
        if not self.video_files:
            messagebox.showwarning("Warning", "No video files uploaded!")
            return

        results = []
        for video in self.video_files:
            result = ml_model(video)
            results.append(result)

        # Сохраняем результаты в таблице
        self.results_table = pd.DataFrame(results)
        messagebox.showinfo("Info", "Processing complete!")

        # Очищаем Treeview перед добавлением новых данных
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Добавляем новые данные в Treeview
        for index, row in self.results_table.iterrows():
            tag = "oddrow" if index % 2 == 0 else "evenrow"
            self.tree.insert("", "end", values=(row["Video"], row["Result"], row["Duration (s)"], row["Count Frame"]), tags=(tag,))

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