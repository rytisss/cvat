# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import tempfile
from pathlib import Path

import numpy as np
import pydicom
from django.test import SimpleTestCase
from PIL import Image
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from cvat.apps.engine.media_extractors import MEDIA_TYPES, DicomReader, _is_dicom, get_mime
from cvat.apps.engine.task import _count_files, _validate_data

XA_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.12.1"


def write_dicom(
    path: Path,
    frames: np.ndarray,
    *,
    bits_stored: int = 8,
    series_uid: str | None = None,
    instance_number: int = 1,
    photometric: str = "MONOCHROME2",
) -> Path:
    """Write a minimal X-ray angiography DICOM file holding `frames` (frames, rows, cols)."""
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[np.newaxis]
    bits_allocated = 8 if bits_stored <= 8 else 16
    dtype = np.uint8 if bits_allocated == 8 else np.uint16

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = XA_IMAGE_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = pydicom.Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = XA_IMAGE_STORAGE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "XA"
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.InstanceNumber = instance_number
    ds.Rows, ds.Columns = frames.shape[1:]
    if frames.shape[0] > 1:
        ds.NumberOfFrames = frames.shape[0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = 0
    ds.PixelData = frames.astype(dtype).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def read_frames(reader: DicomReader) -> list[np.ndarray]:
    frames = []
    for image, _ in reader:
        with Image.open(image) as img:
            frames.append(np.asarray(img))
    return frames


class DicomDetectionTest(SimpleTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_extensionless_dicom_is_detected_by_content(self):
        path = write_dicom(self.tmp / "I1000000", np.zeros((4, 4)))

        self.assertTrue(_is_dicom(path))
        self.assertEqual(get_mime(path), "dicom")

    def test_extensionless_non_dicom_file_is_not_dicom(self):
        path = self.tmp / "VERSION"
        path.write_bytes(b"\x00" * 200)

        self.assertFalse(_is_dicom(path))

    def test_dcm_extension_is_dicom_without_reading_the_file(self):
        self.assertTrue(_is_dicom(self.tmp / "missing.dcm"))

    def test_jpeg_lossless_decoder_is_available(self):
        # IMSP angiograms use JPEG Lossless, Process 14 SV1; pydicom needs a plugin to decode it
        from pydicom.pixels.decoders import JPEGLosslessSV1Decoder

        self.assertTrue(JPEGLosslessSV1Decoder.is_available)


class DicomReaderTest(SimpleTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "src"
        self.src.mkdir()
        self.out = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def make_reader(self, paths):
        return DicomReader(source_paths=list(paths), extract_dir=str(self.out))

    def test_8bit_multiframe_pixels_are_kept_unchanged(self):
        frames = np.stack([np.full((4, 4), 100), np.full((4, 4), 120)])
        frames[0, 0, 0] = 90
        frames[1, 0, 0] = 130
        path = write_dicom(self.src / "File0001.dcm", frames)

        result = read_frames(self.make_reader([path]))

        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[1], frames[1])

    def test_12bit_run_uses_one_window_for_all_frames(self):
        # Frame 0 spans a narrow range, frame 1 the full range. A per-frame stretch would map the
        # same raw value 2000 to different grey levels; one window for the run keeps them equal.
        frame0 = np.full((10, 10), 2000)
        frame0[0, :5] = 1800
        frame1 = np.full((10, 10), 2000)
        frame1[0, :5] = 200
        frame1[0, 5:] = 3800
        path = write_dicom(self.src / "LEO6VP69", np.stack([frame0, frame1]), bits_stored=12)

        result = read_frames(self.make_reader([path]))

        self.assertEqual(result[0].dtype, np.uint8)
        self.assertEqual(result[0][5, 5], result[1][5, 5])
        self.assertLess(result[1][0, 0], result[1][5, 5])
        self.assertGreater(result[1][0, 9], result[1][5, 5])

    def test_monochrome1_is_inverted(self):
        frame = np.array([[0, 255]] * 2)
        path = write_dicom(self.src / "mono1.dcm", frame, photometric="MONOCHROME1")

        (result,) = read_frames(self.make_reader([path]))

        np.testing.assert_array_equal(result, 255 - frame)

    def test_single_frame_series_is_ordered_by_filename_when_instance_numbers_tie(self):
        # UMFCD stores the run number in InstanceNumber, so every frame of a series shares it and
        # the base-36 filename counter (I2000009 < I200000A) is the frame order.
        series = generate_uid()
        names = ["I200000A", "I2000000", "I2000009"]
        values = {"I2000000": 10, "I2000009": 20, "I200000A": 30}
        paths = [
            write_dicom(
                self.src / n, np.full((4, 4), values[n]), series_uid=series, instance_number=2
            )
            for n in names
        ]

        result = read_frames(self.make_reader(paths))

        self.assertEqual([int(f[0, 0]) for f in result], [10, 20, 30])

    def test_single_frame_series_is_ordered_by_instance_number(self):
        series = generate_uid()
        paths = [
            write_dicom(
                self.src / f"frame_{name}",
                np.full((4, 4), value),
                series_uid=series,
                instance_number=number,
            )
            for name, value, number in [("a", 30, 3), ("b", 10, 1), ("c", 20, 2)]
        ]

        result = read_frames(self.make_reader(paths))

        self.assertEqual([int(f[0, 0]) for f in result], [10, 20, 30])

    def test_files_from_different_series_are_rejected(self):
        paths = [
            write_dicom(self.src / "I1000000", np.zeros((4, 4))),
            write_dicom(self.src / "I1000001", np.zeros((4, 4))),
        ]

        with self.assertRaisesRegex(Exception, "2 different series"):
            self.make_reader(paths)

    def test_more_than_one_multiframe_file_is_rejected(self):
        series = generate_uid()
        paths = [
            write_dicom(self.src / "File0001.dcm", np.zeros((2, 4, 4)), series_uid=series),
            write_dicom(self.src / "File0002.dcm", np.zeros((2, 4, 4)), series_uid=series),
        ]

        with self.assertRaisesRegex(Exception, "one multi-frame"):
            self.make_reader(paths)

    def test_source_files_are_kept_when_extract_dir_is_given(self):
        path = write_dicom(self.src / "File0001.dcm", np.zeros((4, 4)))

        self.make_reader([path])

        self.assertTrue(path.exists())


class DicomTaskDataTest(SimpleTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_uploaded_extensionless_dicom_files_are_counted_as_dicom(self):
        series = generate_uid()
        for name in ("I1000000", "I1000001"):
            write_dicom(self.upload_dir / name, np.zeros((4, 4)), series_uid=series)
        data = {"client_files": ["I1000000", "I1000001"], "server_files": [], "remote_files": []}

        counter = _count_files(data, upload_dir=self.upload_dir)

        self.assertEqual(sorted(counter["dicom"]), ["I1000000", "I1000001"])

    def test_several_dicom_files_are_valid_task_data(self):
        counter = {media_type: [] for media_type in MEDIA_TYPES}
        counter["dicom"] = ["I1000000", "I1000001", "I1000002"]

        _, task_mode = _validate_data(counter)

        self.assertEqual(task_mode, "annotation")

    def test_dicom_cannot_be_mixed_with_images(self):
        counter = {media_type: [] for media_type in MEDIA_TYPES}
        counter["dicom"] = ["I1000000"]
        counter["image"] = ["frame.png"]

        with self.assertRaisesRegex(ValueError, "dicom"):
            _validate_data(counter)
