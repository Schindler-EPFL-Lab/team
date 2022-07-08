import tempfile
import unittest
from pathlib import Path

from learning_from_demo.recorders.demonstration_recorder import DemonstrationRecorder


class RecordersTest(unittest.TestCase):
    def test_demonstration_recorder_filename(self):
        tempfile.tempdir = "./tmp"
        tmp_dir = tempfile.gettempdir()
        recorder = DemonstrationRecorder(data_path=tmp_dir)
        file_name = Path(recorder.next_filename).name
        self.assertEqual(file_name, "demonstration_1.json")

        f1 = tempfile.NamedTemporaryFile(suffix=".json", dir="./tmp")  # noqa
        file_name = Path(recorder.next_filename).name
        self.assertEqual(file_name, "demonstration_2.json")

        f2 = tempfile.NamedTemporaryFile(suffix=".json", dir="./tmp")  # noqa
        file_name = Path(recorder.next_filename).name
        self.assertEqual(file_name, "demonstration_3.json")
