import os

from team.demonstration_player import DemonstrationPlayer
from team.trajectory import Trajectory

base_path = os.path.dirname(os.path.dirname(__file__))

if __name__ == "__main__":
    filename = "test_data.json"
    url = "https://localhost:8881"
    file_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "demonstrations"
    )
    filename_path = os.path.join(file_dir, filename)
    play = DemonstrationPlayer(url)
    trajectory = Trajectory.from_file(filename_path)
    play.play(trajectory)
