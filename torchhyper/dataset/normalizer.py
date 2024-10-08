from typing import List, Tuple

from mpire import WorkerPool
import numpy as np


class RunningStats:
    """A class to compute running first- and second-order statistics."""

    def __init__(self,
                 shape: Tuple[int, ...],
                 dtype: np.dtype = np.float32) -> None:
        """
        Initialize a RunningStats object.

        Args:
            shape: A tuple indicating the shape of the input data.
            dtype: The data type used for the computation (default is np.float32).
        """
        self.num_samples = 0
        self.shape = shape
        self.dtype = dtype
        self.running_mean = np.zeros(shape, dtype=dtype)
        self.running_sum_of_differences = np.zeros(shape, dtype=dtype)

    def input_samples(self, samples: np.ndarray, num_workers: int = 8) -> None:
        """Input new samples and update the running stats.

        Args:
            samples: A numpy array of shape (N, *) where N is the number of
                samples and * is consistent with self.shape.
            num_workers: The number of workers used to compute the running
                statistics (default is 8).

        Returns:
            None
        """
        if num_workers > 1:
            # Split the indices into num_workers chunks.
            split_idxs = np.array_split(np.arange(samples.shape[0]),
                                        num_workers,
                                        axis=0)
            with WorkerPool(
                    n_jobs=num_workers,
                    shared_objects=samples,
                    start_method='fork',
            ) as pool:
                # Map the worker function to the indices using a worker pool.
                outputs = pool.map(
                    self.serial_worker,
                    split_idxs,
                    progress_bar=True,
                )
            # Unpack the outputs from the worker pool.
            num_samples, running_mean, running_sum_of_differences = zip(
                *outputs)
            # Update the total number of samples.
            self.num_samples = sum(num_samples)
            # Update the running mean.
            self.running_mean = sum([
                mean / self.num_samples * num_samples
                for mean, num_samples in zip(running_mean, num_samples)
            ])
            # Update the running sum of squared differences.
            self.running_sum_of_differences += sum(
                [sum_of_diff for sum_of_diff in running_sum_of_differences])
        else:
            # Compute the running statistics using a single worker.
            (self.num_samples, self.running_mean,
             self.running_sum_of_differences) = self.serial_worker(
                 samples, range(samples.shape[0]))

    def serial_worker(
            self, samples: np.ndarray,
            split_idxs: List[int]) -> Tuple[int, np.ndarray, np.ndarray]:
        """Compute the running stats using a single worker.

        Args:
            samples: A numpy array of shape (N, *) where N is the number of
                samples and * represents any number of dimensions.
            split_idxs: A list of indices to process.

        Returns:
            A tuple containing the number of processed samples, the running
            mean, and the running sum of squared differences.
        """
        num_samples = 0
        running_mean = np.zeros(self.shape, dtype=self.dtype)
        running_sum_of_differences = np.zeros(self.shape, dtype=self.dtype)
        for i in split_idxs:
            # Update the running mean.
            num_samples += 1
            delta = samples[i, ...] - running_mean
            running_mean = running_mean + delta / num_samples
            # Update the running sum of squared differences.
            delta2 = samples[i, ...] - running_mean
            running_sum_of_differences += delta * delta2
        return num_samples, running_mean, running_sum_of_differences

    def compute_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the weighted mean and standard deviation.

        Returns:
            Tuple of two NumPy arrays, where the first element is the running
            mean and the second element is the standard deviation.
        """
        # Compute the standard deviation based on the intermediate weighted sum
        # and sum of squares.
        std = np.sqrt(
            np.sum(self.running_sum_of_differences) / self.num_samples /
            np.prod(self.running_sum_of_differences.shape))

        # Return the running mean and standard deviation as a tutorchhyper.
        return np.mean(self.running_mean), std


class Normalizer:
    """Normalizer a tensor image with training mean and standard deviation.
    Extracts the mean and standard deviation from the training dataset, and
    uses them to normalize an input image.
    """

    def __init__(self,
                 mean: np.ndarray,
                 std: np.ndarray,
                 eps: float = 1e-6) -> None:
        """Initializes a Normalizer object.

        Args:
            mean: A numpy array that contains the mean of input data.
            std: A numpy array that contains the standard deviation of input
                dimension.
            eps: An optional small float to avoid dividing by 0.
        """
        # Compute the training dataset mean and standard deviation over the
        # batch dimensions.
        self.mean = mean
        self.std = std
        self.eps = eps

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply normalization to input samtorchhyper.

        Args:
            x: A numpy array with the same dimension organization as `dataset`.

        Returns:
            A numpy array with the same dimension organization as `x` but
            normalized with the mean and standard deviation of the training
            dataset.
        """
        return (x - self.mean) / (self.std + self.eps)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """Restore the normalization from the input samtorchhyper.

        Args:
            x: A normalized numpy array with the same dimension organization as
                `dataset`.

        Returns:
            A numpy array with the same dimension organization as `x` that has
            been unnormalized.
        """
        return x * (self.std + self.eps) + self.mean
