import tkinter as tk


root = tk.Tk()
root.title("Attack Detected")
#Creates the main window (root) titled "Attack Detected".

frame = tk.Frame(root)
frame.grid(row=0, column=0, sticky="nsew")

root.geometry("600x400")
#Sets a fixed window size of 600x400 pixels.

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
#Makes the grid responsive: columns and rows expand equally with the window.
#weight=1 ensures resizing distributes space proportionally.

def on_click():
    exit()

lbl = tk.Label(root, text="Is there currently an obstruction?")
lbl.grid(row=0, column=0, columnspan=2, sticky="s")
lbl.config(font=("Arial", 30))
#Displays a large question label at the top center.
#columnspan=2 makes it span across both button columns.

btnYes = tk.Button(root, text="Yes", command=on_click)
btnYes.grid(row=1, column=0, sticky="ne", padx=50, pady=30)
btnYes.config(font=("Arial", 20))

btnNo = tk.Button(root, text="No", command=on_click)
btnNo.grid(row=1, column=1, sticky="nw", padx=50, pady=30)
btnNo.config(font=("Arial", 20))
#Two buttons: Yes and No.
#Both trigger the same function on_click() when pressed.


root.mainloop()
