import os

from arco.learning_from_demo.play_demo import PlayBack

base_path = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    filename = "robot_trajectory_cf1.json"
    url = "https://localhost:8881"
    file_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests"
    )
    filename_path = os.path.join(file_dir, filename)
    play = PlayBack(filename_path=filename_path, base_url=url)
    play.play()
