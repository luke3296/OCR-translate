import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import OCR
import numpy as np


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        self.OCR = OCR.Translator()

        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.root.bind("<Key>", self.handle_key)

        self.rect_color = "red"
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.pending_rect = None
        self.selected_regions = []

        self.image = None
        self.photo = None

        self.create_menu()

    def create_menu(self):
        menubar = tk.Menu(self.root)

        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open_image)
        filemenu.add_command(label="Save", command=self.save_image)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # process dropdown
        processmenu = tk.Menu(menubar, tearoff=0)
        processmenu.add_command(label="Process", command=self.process_selected_regions)
        menubar.add_cascade(label="Process", menu=processmenu)
        # Language dropdown menu
        language_menu = tk.Menu(menubar, tearoff=0)
        self.language = tk.StringVar(value="jp")  # Default language is Japanese

        def set_language_jp():
            self.language.set("jp")

        def set_language_cn():
            self.language.set("cn")

        language_menu.add_radiobutton(label="Japanese", variable=self.language, value="jp", command=set_language_jp)
        language_menu.add_radiobutton(label="Chinese", variable=self.language, value="cn", command=set_language_cn)

        menubar.add_cascade(label="Language", menu=language_menu)

        self.root.config(menu=menubar)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            self.canvas.delete("rectangle")
            self.canvas.delete("text")
            self.selected_regions=[]
            
            self.image = Image.open(file_path)
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            

    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_mouse_wheel(self, event):
        self.canvas.yview_scroll(-int(event.delta / 120), "units")

    def on_mouse_down(self, event):
        if self.image:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            self.start_x = x
            self.start_y = y
            self.rect = self.canvas.create_rectangle(
                x,
                y,
                x,
                y,
                outline=self.rect_color,
                tags="rectangle",
            )

    def on_mouse_move(self, event):
        if self.rect:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.rect, self.start_x, self.start_y, x, y)

    def on_mouse_up(self, event):
        if self.rect and self.image:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            end_x, end_y = x, y
            x1, y1, x2, y2 = self.canvas.coords(self.rect)
            if x1 != x2 and y1 != y2:
                self.pending_rect = self.rect

    def handle_key(self, event):
        if event.keysym == "Return" and self.pending_rect is not None:
            self.canvas.itemconfig(self.pending_rect, outline="blue")
            self.selected_regions.append(self.pending_rect)
            self.rect = None
            self.pending_rect = None

    def save_image(self):
        if self.image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("GIF", "*.gif")],
            )
            if file_path:
                self.image.save(file_path)

    def overlay_text_on_image(self, rect_coords, text):
        # Create a copy of the original image to avoid modifying it directly
        image_with_overlay = self.image.copy()

        # Convert the image to RGB mode
        image_with_overlay = image_with_overlay.convert("RGB")
        draw = ImageDraw.Draw(image_with_overlay)

        # Extract rectangle coordinates
        x1, y1, x2, y2 = rect_coords

        # Draw a new rectangle
        rect_color = tuple(image_with_overlay.getpixel((x1, y1)))  # Convert to tuple
        rect_color_with_alpha = rect_color + (128,)  # Add transparency to the color
        draw.rectangle([(x1, y1), (x2, y2)], fill=rect_color_with_alpha)

        # Write the text on the new rectangle
        font = ImageFont.truetype("arial.ttf", size=14)
        text_color = (0, 0, 0)  # Black color for text
        text_position = (x1 + 5, y1 + 5)  # Offset the text position
        draw.text(text_position, text, font=font, fill=text_color)

        # Update the canvas with the modified image
        self.photo = ImageTk.PhotoImage(image_with_overlay)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        return image_with_overlay

    def process_selected_regions(self):
        extracted_texts = []
        if self.selected_regions:
            print("Selected Regions:", len(self.selected_regions))
            for rect in self.selected_regions:
                x1, y1, x2, y2 = map(int, self.canvas.coords(rect))
                cropped_image = self.image.crop((x1, y1, x2, y2)).convert("L")
                img_data = np.array(cropped_image)
                img_data = img_data[:, :, 0] if img_data.ndim == 3 else img_data
                img_data = np.squeeze(img_data)

                if self.language.get() == "jp":
                    extracted_text = self.OCR.extract_text_jp(img_data)
                else:
                    extracted_text = self.OCR.extract_text_cn(img_data)

                extracted_texts.append(extracted_text)
                print(f"X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}")

                # Overlay extracted text on the original image
                self.image = self.overlay_text_on_image((x1, y1, x2, y2), extracted_text)

        else:
            print("No regions selected.")

        print("Extracted Texts:")
        for text in extracted_texts:
            print(text)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
