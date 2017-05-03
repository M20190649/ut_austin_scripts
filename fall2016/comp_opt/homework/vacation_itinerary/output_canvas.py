import sys
if sys.version_info[0] < 3:
	import Tkinter as tk
else:
	import tkinter as tk
from PIL import Image, ImageTk

master = tk.Tk()
master.title('Optimal Itinerary')



# Works only if there is a single display; otherwise sums the value from each display
screen_width = master.winfo_screenwidth()
screen_height = master.winfo_screenheight()
master.maxsize(screen_width, int(0.9 * screen_height))


number_frame = 3	# number of frames
frame_width = int(0.8 * screen_width) / number_frame
frame_height = int(0.8 * screen_height)
master.minsize(frame_width, int((9 * frame_width) / 16))

def AddFrame(parentwidget, text, width = frame_width, height = frame_height, relief = tk.SUNKEN):
	return tk.LabelFrame(parentwidget, text = text, height = height, width = width, relief = relief, bd = 3, font = 'Literata 20')

# Create three frames
map_frame = AddFrame(master, text = 'Map')
# map_frame.pack(side = tk.LEFT, fill = tk.BOTH, padx = 5, pady = 5)
map_frame.grid(row = 0, column = 0, padx = 5, pady = 5, sticky = tk.N)
map_frame_label = tk.Label(map_frame, text = 'Map:', font = 'Literata 20', justify = tk.CENTER, bd = 0, cursor = 'spider')
# map_frame_label.pack(side = tk.TOP, fill = tk.X)
map_frame_label.grid(row = 0, column = 0, sticky = tk.W + tk.E + tk.N)
# map_frame_label.place(map_frame)

itinerary_frame = AddFrame(master, text = 'Schedule')
# itinerary_frame.pack(side = tk.LEFT)
itinerary_frame.grid(row = 0, column = 1)

# Add element of itinerary_frame
itinerary_frame_label = tk.Label(itinerary_frame, text = 'Schedule:', font = 'Literata 20', justify = tk.CENTER, bd = 0, cursor = 'spider')
# itinerary_frame_label.pack(side = tk.TOP, fill = tk.X)
itinerary_frame_label.grid(row = 0, column = 0, sticky = tk.W + tk.E)
itinerary_frame_list = tk.Listbox(itinerary_frame)
# Add scrollbars
itinerary_yscrollbar = tk.Scrollbar(itinerary_frame_list)
# itinerary_yscrollbar.pack(side = tk.RIGHT, fill = tk.BOTH)
itinerary_yscrollbar.grid(row = 1, column = 1, sticky = tk.W)
itinerary_frame_list.yscrollcommand = itinerary_yscrollbar.set
# itinerary_frame_list.pack(side = tk.LEFT, fill = tk.BOTH)
itinerary_frame_list.grid(row = 1, column = 0)

itinerary_yscrollbar.config(command = itinerary_frame_list.yview)

review_frame = AddFrame(master, text = 'Reviews')
# review_frame.pack(side = tk.LEFT)
review_frame.grid(row = 0, column = 2)


# Add elements of itinerary





# Add image to the first frame
image = Image.open('ToRudys.png')
width, height = image.size
map_width = frame_width
map_height = (map_width * height) / width
resized_image = image.resize((map_width, map_height), Image.ANTIALIAS)
route_map = ImageTk.PhotoImage(resized_image)
image_label = tk.Label(map_frame, image = route_map, relief = tk.SUNKEN)
image_label.image = route_map
# image_label.pack()
image_label.grid(row = 1, column = 0)


# Add button to remove everything save the map
def ZoomMap():
	itinerary_frame.destroy()
	review_frame.destroy()
	image = Image.open('ToRudys.png')
	resized_image = image.resize((1024, frame_height), Image.ANTIALIAS)
	route_map = ImageTk.PhotoImage(resized_image)
	# image_label = tk.Label(map_frame, image = route_map, relief = tk.SUNKEN)
	image_label['image'] = route_map
	image_label.image = route_map

map_frame_button = tk.Button(map_frame, text = 'Zoom', font = '16', command = ZoomMap)
# map_frame_button.pack(padx = 3, pady = 3)
map_frame_button.grid(row = 2, column = 0)

# # Create Menu items
# menu = tk.Menu(master)
# master.config(menu = menu)	# attach menu to master

# # Menu operations
# def CmdNewSearch():
# 	print 'New Search'

# def CmdPrint():
# 	print 'Itinerary'

# def CmdExit():
# 	master.destroy()

# def HelpAbout():
# 	print 'Version 1.0'

# def MailAuthor(name):
# 	import webbrowser
# 	email = {'Melissa': 'mmlee246@gmail.com', 'Paul': 'pjruess@gmail.com', 'Sudesh': 'sudesh@utexas.edu'}[name]
# 	link = 'mailto:{0}?subject={1}&body=Hi%20{2}\n'.format(email, 'Help%20with%20Optimal%20Itinerary', name)
# 	webbrowser.open(link)

# # Commands Menu
# filemenu = tk.Menu(menu)
# menu.add_cascade(label = 'Commands', menu = filemenu)
# filemenu.add_command(label = 'New Search', command = CmdNewSearch)
# filemenu.add_command(label = 'Print Itinerary', command = CmdPrint)
# filemenu.add_separator()
# filemenu.add_command(label = 'Exit', command = CmdExit)

# # Help Menu
# helpmenu = tk.Menu(menu)
# menu.add_cascade(label = 'Help', menu = helpmenu)
# helpmenu.add_command(label = 'About', command = HelpAbout)

# # Authors
# authormenu = tk.Menu(helpmenu)
# helpmenu.add_cascade(label = 'Email authors', menu = authormenu)
# authormenu.add_command(label = 'Melissa Lee', command = lambda: MailAuthor('Melissa'))
# authormenu.add_command(label = 'Paul Ruess', command = lambda: MailAuthor('Paul'))
# authormenu.add_command(label = 'Sudesh Agrawal', command = lambda: MailAuthor('Sudesh'))



# Place the widgets on the master using grid



# review_frame = tk.Frame(master, bd = 2, relief = tk.SUNKEN)
# review_frame.grid_rowconfigure(0, weight = 1)	# configure row index of a grid
# review_frame.grid_columnconfigure(0, weight = 1)

# # vertical scrollbar
# # vscrollbar = tk.Scrollbar(master)
# # vscrollbar.pack(side = tk.RIGHT, fill = tk.Y)
# vscrollbar = tk.Scrollbar(review_frame)
# vscrollbar.grid(row = 0, column = 1, sticky = tk.N + tk.S)

# # # horizontal scrollbar
# # hscrollbar = tk.Scrollbar(master, orient = tk.HORIZONTAL)
# # hscrollbar.pack(side = tk.BOTTOM, fill = tk.X)

# # wrap: WORD, CHAR, NONE
# # review_text = tk.Text(master, wrap = tk.WORD, yscrollcommand = vscrollbar.set)
# # review_text.pack()
# # # Alternatively for wrap = NONE
# # review_text = tk.Text(master, wrap = tk.NONE, xscrollcommand = hscrollbar.set, yscrollcommand = vscrollbar.set)
# # review_text.pack()
# review_text = tk.Text(review_frame, wrap = tk.NONE, bd = 0, yscrollcommand = vscrollbar.set)
# review_text.grid(row = 0, column = 0, sticky = tk.N+tk.S+tk.E+tk.W)

# review_text.insert(tk.END, 'Hello World!\nYeah')
# # hscrollbar.config(command = review_text.xview)
# vscrollbar.config(command = review_text.yview)
# review_frame.pack()

master.mainloop()