import threading

import keyboard

from team.recorders.lead_demonstration_recorder import RecordTask

path_to_store = "sample_test.json"
url = "https://localhost:8881"


if __name__ == "__main__":

    record_task = RecordTask(url, data_path=path_to_store)
    t = threading.Thread(target=record_task.run, args=(10,))
    t.start()
    while True:
        if keyboard.is_pressed("q"):
            break

    record_task.terminate()
    t.join()
