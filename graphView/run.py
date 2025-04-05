import webbrowser, os, time
import threading

def server():
    os.system("python -m http.server")

threading.Thread(target=server).start()
time.sleep(1.0)
webbrowser.open("localhost:8000")
