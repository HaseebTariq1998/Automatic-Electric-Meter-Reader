import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import ImageTk, Image
import cv2
from pymongo import MongoClient
import datetime
from PythonScripts.meter_reading_detector import process_video

filename = ""
frame = None
processing_icon = cv2.imread("UiAssets/processing_icon.png")
meternoo = ""
meter_reading = ""
client = MongoClient('localhost', 27017)


def browse_files():
    global filename, frame
    filename = filedialog.askopenfilename(
        initialdir="D:\\INTERNSHIP\\cnn",
        title="Select a File",
        filetypes=[("all video format", ".mp4"),
                   ("all video format", ".flv"),
                   ("all video format", ".avi")]
    )

    label_file_explorer.configure(text="File Opened: " + filename)
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    dsize = (507, 400)
    img = cv2.resize(frame, dsize)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)
    label.configure(image=photo)
    label.image = photo
    button_start.configure(bg="green")
    cap.release()


def process():
    global meternoo, meter_reading, frame
    if filename:
        button_start.configure(text="processing ", bg="red")
        l1.configure(text="Meter No :NA")
        l2.configure(text="Reading :NA")

        dsize = (507, 400)
        img = cv2.resize(frame, dsize)
        img[136:264, 189:317] = processing_icon
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)

        label.configure(image=photo)
        label.image = photo

        window.update_idletasks()
        img, meternoo, meter_reading = process_video(filename)
        l1.configure(text="Meter No :" + meternoo)
        l2.configure(text="Reading :" + meter_reading)
        dsize = (507, 400)
        img = cv2.resize(img, dsize)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)

        label.configure(image=photo)
        label.image = photo
        button_start.configure(text="Start ", bg="green")
    else:
        messagebox.showwarning("showwarning", "No file selected")


def search():
    try:
        global T
        global l5
        s = T.get("1.0", "end")
        d = client["Auto_meter_reading"]
        c = d["MetersReading"]
        myquery = {"MeterNo": int(s)}
        mydoc = c.find(myquery)
        string = ""
        for x in mydoc:
            string = string + "\n   date :" + x["Date"] + "   meter no :" + str(x["MeterNo"]) + "   reading :" + str(
                x["Reading"])
        l5.configure(text=string)
    except:
        messagebox.showerror("Error", "Database is not connected")


def open_new_window():
    global T, l5
    new_window = tk.Toplevel(window)
    new_window.title("Searching Database")
    new_window.geometry("400x400")

    button_search = tk.Button(new_window, text="Search", command=search)
    button_search.grid(row=1, column=1)

    T = tk.Text(new_window, height=1, width=50)
    T.grid(row=1, column=2)

    lf2 = ttk.LabelFrame(new_window, text="< R E S U L T S >", height=50, width=100)
    lf2.grid(row=2, column=1, columnspan=2, sticky='nwse')

    l5 = tk.Label(lf2, text="NA ", height=10, width=50)
    l5.grid(row=2, column=1, columnspan=2, sticky='nwse')


def change_meter_no():
    global meternoo
    meternoo = simpledialog.askstring("Input", "Enter New meter No= ?")
    l1.configure(text="Meter No:" + meternoo)


def change_meter_reading():
    global meter_reading
    meter_reading = simpledialog.askstring("Input", "Enter New meter Reading= ?")
    l2.configure(text="Reading :" + meter_reading)


def insert_record():
    try:
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d %H:%M:%S")
        d = client["Auto_meter_reading"]
        c = d["MetersReading"]
        mydict = {"Date": date, "MeterNo": int(meternoo), "Reading": float(meter_reading)}
        c.insert_one(mydict)
        messagebox.showinfo("showinfo", "Result submitted !!!")
    except:
        messagebox.showerror("Error", "Database is not connected")


# Create the root window
window = tk.Tk()

# Set window title
window.title('Smart Meter Reading ')

# Set window size
window.geometry("1260x469")

label_file_explorer = tk.Label(
    window,
    text="Choose file from device",
    width=60, height=2,
    fg="blue", relief="ridge", borderwidth=2, bg="gray87"
)

button_explore = tk.Button(
    window,
    text="Browse Files",
    command=browse_files, width=30, height=2, bg="gray87"
)

button_exit = tk.Button(
    window,
    text="Submit ",
    command=insert_record, width=40, height=2, bg="gray87"
)

button_1 = tk.Button(
    window,
    text="Edit Meter No",
    command=change_meter_no, width=40, height=2, bg="gray87"
)

button_2 = tk.Button(
    window,
    text="Edit Meter reading ",
    command=change_meter_reading, width=40, height=2, bg="gray87"
)

button_start = tk.Button(
    window,
    text="Start",
    command=process, width=71, height=2, bg="gold3"
)

button_search = tk.Button(
    window,
    text="Search",
    command=open_new_window, width=40, height=2, bg="gray87"
)

img = cv2.imread("UiAssets/upload_icon.PNG")
scale_percent = 15

width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)

dsize = (500, 400)
img = cv2.resize(img, dsize)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
photo = ImageTk.PhotoImage(image)

lf = labelFrame = ttk.LabelFrame(window, text="< R E S U L T S >", height=50, width=100)
lf.grid(column=1, row=3, columnspan=2, padx=20, pady=40, sticky="news")

l1 = tk.Label(lf, text="Meter No: NA", width=40, height=2)
l2 = tk.Label(lf, text="Readinng : NA", width=40, height=2)

label = tk.Label(image=photo, borderwidth=3, relief="solid")
label.image = photo
label.grid(column=3, row=1, rowspan=10, padx=5, pady=5)

label_file_explorer.grid(column=2, row=2, padx=5, pady=5)
button_explore.grid(column=1, row=2, padx=5, pady=5, sticky=tk.E)
button_exit.grid(column=1, row=5, padx=5, pady=10)
button_1.grid(column=1, row=4, padx=5, pady=5)
button_2.grid(column=2, row=4, padx=5, pady=5)
button_start.grid(column=3, row=11)
button_search.grid(column=2, row=5, padx=5, pady=5)

l1.grid(row=3, column=1, padx=5, pady=5)
l2.grid(row=3, column=2, padx=5, pady=5)

window.mainloop()
