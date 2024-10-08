import numpy as np
import unittest
from torchhyper.dataset import RunningStats


class TestRunningStats(unittest.TestCase):
    """
    A test class for the RunningStats class.
    """

    def setUp(self) -> None:
        """
        Set up test parameters.
        """
        self.shape: tuple[int, int] = (2, 3)
        self.dtype: type = np.float32

    def test_init(self) -> None:
        """
        Test initialization of RunningStats.
        """
        rs: RunningStats = RunningStats(self.shape, self.dtype)
        self.assertEqual(rs.num_samples, 0)
        self.assertEqual(rs.shape, self.shape)
        self.assertEqual(rs.dtype, self.dtype)
        self.assertTrue(
            np.all(rs.running_mean == np.zeros(
                self.shape,
                dtype=self.dtype,
            )))
        self.assertTrue(
            np.all(rs.running_sum_of_differences == np.zeros(
                self.shape,
                dtype=self.dtype,
            )))

    def test_input_samples(self) -> None:
        """
        Test input_samples method of RunningStats.
        """
        rs: RunningStats = RunningStats(self.shape, self.dtype)
        samples: np.ndarray = np.array(
            [[1, 2, 3], [4, 5, 6]],
            dtype=self.dtype,
        ).reshape((1, 2, 3))
        samples: np.ndarray = np.repeat(samples, 100, axis=0)
        rs.input_samples(samples)
        self.assertEqual(rs.num_samples, 100)
        self.assertTrue(
            np.allclose(
                rs.running_mean,
                np.array([[1., 2., 3.], [4., 5., 6.]], dtype=self.dtype),
            ))
        self.assertTrue(
            np.allclose(
                rs.running_sum_of_differences,
                np.array([[0., 0., 0.], [0., 0., 0.]], dtype=self.dtype),
            ))

    def test_compute_stats(self) -> None:
        """
        Test compute_stats method of RunningStats.
        """
        for num_workers in [1, 8]:
            for batchsize in [25, 100]:

                rs: RunningStats = RunningStats(self.shape, self.dtype)

                samples: np.ndarray = np.array(
                    [[1, 2, 3], [4, 5, 6]],
                    dtype=self.dtype,
                ).reshape((1, 2, 3))

                samples: np.ndarray = np.repeat(samples, 100, axis=0)

                for i in range(0, samples.shape[0], batchsize):
                    batch: np.ndarray = samples[i:i + batchsize, ...]
                    rs.input_samples(batch, num_workers=num_workers)
                mean, std = rs.compute_stats()

                self.assertTrue(
                    np.allclose(
                        mean,
                        np.mean(
                            np.array(
                                [[1., 2., 3.], [4., 5., 6.]],
                                dtype=self.dtype,
                            ))))
                self.assertTrue(np.allclose(
                    std,
                    0.0,
                ))


if __name__ == '__main__':
    unittest.main()
