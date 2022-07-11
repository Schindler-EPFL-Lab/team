from timeit import default_timer as timer

from learning_from_demo.recorders.demonstration_recorder import DemonstrationRecorder
from rws2.RWS_wrapper import RwsWrapper


class LeadDemonstrationRecorder(DemonstrationRecorder):
    def __init__(self, robot_url: str, data_path: str) -> None:
        """Initializes the recording session. Sets the demonstration name and puts the
        robot in gravity compensation mode"""
        super().__init__(data_path=data_path)

        # Robot address and Robot Web Services connection
        self.url = robot_url
        self.rws = RwsWrapper(robot_url=self.url)

        # Setup robot mode
        self.rws.activate_lead_through()

        # Since RWS2 doesn't return a timestamp with the measurement,
        # compute it from Python
        self.t_start = timer()

    def record(self) -> None:
        """Records one snapshot of the robot data"""

        # Read data from robot
        tcp_pos, tcp_ori, rob_cf = self.rws.robot.get_tcp_info()
        joints = self.rws.robot.get_joints_positions()
        timestamp = [timer() - self.t_start]
        values_list = timestamp + tcp_pos + tcp_ori + rob_cf + joints
        # Check that all information are available
        if tcp_pos and tcp_ori and rob_cf and joints:
            info = {key: value for (key, value) in zip(self.data.keys(), values_list)}
            self.update(tmp_dict=info)

    def stop_and_save_recording(self) -> None:
        """Stops and saves recording"""

        # Robot shutdown operations
        self.rws.deactivate_lead_through()
        # Save data to file
        self.create_file()


class RecordTask:
    """Recording task that can be started and stoped."""

    def __init__(self, robot_url: str, data_path: str):
        self._running = True
        self.record = LeadDemonstrationRecorder(
            robot_url=robot_url, data_path=data_path
        )

    def terminate(self):
        self._running = False
        self.record.stop_and_save_recording()

    def run(self, n: int):
        while self._running:
            self.record.record()
