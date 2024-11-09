import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import cv2

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
        
        self.video_files = []
        
        # Кнопка для загрузки видео
        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(pady=10)
        
        # Кнопка для обработки видео
        self.process_button = tk.Button(root, text="Process Video", command=self.process_video)
        self.process_button.pack(pady=10)
        
        # Кнопка для сохранения результатов
        self.save_button = tk.Button(root, text="Save Results", command=self.save_results)
        self.save_button.pack(pady=10)
        
        # Таблица результатов
        self.results_table = None
        
    def upload_video(self):
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
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
        
        self.results_table = pd.DataFrame(results)
        messagebox.showinfo("Info", "Processing complete!")
        print(self.results_table)  # Для отладки, можно убрать
    
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
    root.geometry("800x600+900+500")
    app = VideoProcessingApp(root)
    root.mainloop()
