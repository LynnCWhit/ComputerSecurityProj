import tkinter as tk


root = tk.Tk()
root.title("Attack Detected")

frame = tk.Frame(root)
frame.grid(row=0, column=0, sticky="nsew")

root.geometry("600x400")

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)


def on_click():
    exit()

lbl = tk.Label(root, text="Is there currently an obstruction?")
lbl.grid(row=0, column=0, columnspan=2, sticky="s")
lbl.config(font=("Arial", 30))

btnYes = tk.Button(root, text="Yes", command=on_click)
btnYes.grid(row=1, column=0, sticky="ne", padx=50, pady=30)
btnYes.config(font=("Arial", 20))

btnNo = tk.Button(root, text="No", command=on_click)
btnNo.grid(row=1, column=1, sticky="nw", padx=50, pady=30)
btnNo.config(font=("Arial", 20))


root.mainloop()