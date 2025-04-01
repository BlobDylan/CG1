import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
from typing import List, Tuple
import functools


def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)

    return wrap_fn


class SeamImage:
    def __init__(self, img_path: str, vis_seams: bool = True):
        """SeamImage initialization.

        Parameters:
            img_path (str): image local path
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path

        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T

        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()

        self.h, self.w = self.rgb.shape[:2]

        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))
        self.idx_map = np.stack([self.idx_map_h, self.idx_map_v], axis=2)

    @NI_decor
    def rgb_to_grayscale(self, np_img):
        """Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3)
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        return (np_img @ self.gs_weights).squeeze()

    @NI_decor
    def calc_gradient_magnitude(self):
        """Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        padded = np.pad(
            self.resized_gs,
            ((0, 1), (0, 1)),
            mode="constant",
            constant_values=0.5,
        )
        return np.sqrt(
            np.square(np.diff(padded, axis=0)[:, :-1])
            + np.square(np.diff(padded, axis=1)[:-1, :])
        )

    def update_ref_mat(self):
        for i, s in enumerate(self.seam_history[-1]):
            self.idx_map[i, s:] += 1

    def reinit(self, img_path):
        """
        Re-initiates instance and resets all variables.
        """
        self.__init__(img_path= img_path if img_path else self.path)

    @staticmethod
    def load_image(img_path, format="RGB"):
        return (
            np.asarray(Image.open(img_path).convert(format)).astype("float32") / 255.0
        )

    def paint_seams(self):
        self.cumm_mask = np.zeros_like(self.rgb, dtype=bool)
        self.cumm_mask[self.idx_map[:self.h, :self.w, 1], self.idx_map[:self.h, :self.w,0]] = True
        self.seams_rgb = np.where(self.cumm_mask, self.rgb, [1,0,0])

    def seams_removal(self, num_remove: int):
        """Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seams to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, mask) where:
                - E is the gradient magnitude matrix
                - mask is a boolean matrix for removed seams
            iii) find the best seam to remove and store it
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the chosen seam (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you wish, but it needs to support:
            - removing seams a couple of times (call the function more than once)
            - visualize the original image with removed seams marked in red (for comparison)
        """

        for _ in tqdm(range(num_remove)):
            self.E = self.calc_gradient_magnitude()
            self.mask = np.ones_like(self.E, dtype=bool)

            seam = self.find_minimal_seam()
            self.seam_history.append(seam)
            # if self.vis_seams:
            #     self.update_ref_mat()
            self.remove_seam(seam)
        
        self.paint_seams()
        

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the seam with the minimal energy.
        Returns:
            The found seam, represented as a list of indexes
        """
        raise NotImplementedError(
            "TODO: Implement SeamImage.find_minimal_seam in one of the subclasses"
        )

    @NI_decor
    def remove_seam(self, seam: List[int]):
        """
        Removes a seam from self.resized_rgb and self.resized_gs.
        Updates idx_map_h and idx_map_v.
        If self.vis_seams is True, paints the removed seam in red on seams_rgb (original size).
        """
        h = self.resized_rgb.shape[0]

        # Build boolean mask to remove the seam
        mask = np.ones((h, self.resized_rgb.shape[1]), dtype=bool)
        for i in range(h):
            mask[i, seam[i]] = False

        # Resize RGB and grayscale images
        self.resized_rgb = self.resized_rgb[mask].reshape((h, -1, 3))
        self.resized_gs = self.resized_gs[mask].reshape((h, -1))

        # If enabled, mark the removed seam in red on the original seams_rgb image
        if self.vis_seams:
            # for i in range(h):
            #     y, x = self.idx_map_v[i, seam[i]], self.idx_map_h[i, seam[i]]
            #     self.seams_rgb[y, x] = [1.0, 0.0, 0.0]  # red
            self.idx_map = self.idx_map[mask].reshape((h, -1, 2))

        # Update image width
        self.w -= 1


    @NI_decor
    def rotate_mats(self, clockwise: bool):
        """
        Rotates the matrices either clockwise or counter-clockwise.
        """
        mat = (1,0) if clockwise else (0,1)
        self.resized_gs = np.rot90(self.resized_gs, k=1, axes=mat)
        self.resized_rgb = np.rot90(self.resized_rgb, k=1, axes=mat)
        self.h , self.w = self.w, self.h

        if self.vis_seams:
            self.idx_map = np.rot90(self.idx_map, k=1, axes=mat)

    @NI_decor
    def seams_removal_vertical(self, num_remove: int):
        """A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.seams_removal(num_remove)

    @NI_decor
    def seams_removal_horizontal(self, num_remove: int):
        """Removes num_remove horizontal seams by rotating the image, removing vertical seams, and restoring the original rotation.

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        self.rotate_mats(True)
        self.seams_removal(num_remove)
        self.rotate_mats(False)

    """
    BONUS SECTION
    """

    @NI_decor
    def seams_addition(self, num_add: int):
        """BONUS: adds num_add seams to the image

        Parameters:
            num_add (int): number of horizontal seam to be removed

        Guidelines & hints:
        - This method should be similar to removal
        - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
        - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition")

    @NI_decor
    def seams_addition_horizontal(self, num_add: int):
        """A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_add (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError(
            "TODO (Bonus): Implement SeamImage.seams_addition_horizontal"
        )

    @NI_decor
    def seams_addition_vertical(self, num_add: int):
        """A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError(
            "TODO (Bonus): Implement SeamImage.seams_addition_vertical"
        )


class GreedySeamImage(SeamImage):
    """Implementation of the Seam Carving algorithm using a greedy approach"""

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using a greedy algorithm.

        Guidelines & hints:
        The first pixel of the seam should be the pixel with the lowest cost.
        Every row chooses the next pixel based on which neighbor has the lowest cost.
        """
        # raise NotImplementedError("TODO: Implement GreedySeamImage.find_minimal_seam")
        h, w = self.E.shape
        seam = np.zeros(h, dtype=int)
        min_idx = np.argmin(self.E[0])
        for i in range(h):
            seam[i] = min_idx
            left_idx = max(min_idx - 1, 0)
            right_idx = min(min_idx + 1, w - 1)
            if i + 1 < h:
                min_idx = np.argmin(self.E[i + 1][left_idx : right_idx + 1]) + left_idx
        return seam.tolist()


class DPSeamImage(SeamImage):
    """
    Implementation of the Seam Carving algorithm using dynamic programming (DP).
    """

    def __init__(self, *args, **kwargs):
        """DPSeamImage initialization."""
        super().__init__(*args, **kwargs)
        try:
            self.init_mats()

        except NotImplementedError as e:
            print(e)

    @NI_decor
    def find_minimal_seam(self) -> List[int]:
        """
        Finds the minimal seam by using dynamic programming.

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (M, backtracking matrix) where:
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
        """
        self.init_mats()
        return self.backtrack()
        
    #TODO
    def backtrack(self) -> List[int]:
        min_idx = np.argmin(self.M[-1, :])
        seam = [min_idx]
        for i in range(self.h - 2, -1, -1):
            L_V_or_R = self.backtrack_mat[i+1, min_idx]
            min_idx += L_V_or_R
            min_idx = max(0, min(min_idx, self.w - 2))

            seam.append(min_idx)
            
        seam.reverse()
        return seam
    
    def calc_C(self):
        j_plus = np.roll(self.resized_gs, shift=-1, axis=0)
        j_minus = np.roll(self.resized_gs, shift=1, axis=0)
        i_minus = np.roll(self.resized_gs, shift=1, axis=1)

        self.C_V = np.abs(j_plus - j_minus)
        self.C_L = np.abs(j_plus - j_minus) + np.abs(j_minus - i_minus)
        self.C_R = np.abs(j_plus - j_minus) + np.abs(j_plus - i_minus)
        

    @NI_decor
    def calc_M(self):
        """Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        h, w = self.E.shape
        M = self.E.copy()
        for i in range(1,h):
            j = M[i-1]
            j_plus = np.roll(j, shift=-1, axis=0)
            j_minus = np.roll(j, shift=1, axis=0)

            j += self.C_V[i-1]
            j_minus += self.C_L[i-1]
            j_plus += self.C_R[i-1]

            stacked = np.stack([j_minus, j, j_plus], axis=1)
            directions = np.argmin(stacked, axis=1)
            self.backtrack_mat[i, :] = directions.squeeze()
            min_values = np.min(stacked, axis=1)
            M[i, :] = M[i, :] + min_values.squeeze()
        
        # Changing the range of indexes to -1, 0, 1 (L,V,R)
        self.backtrack_mat -= 1
        return M

    def init_mats(self):
        self.backtrack_mat = np.zeros_like(self.E, dtype=int)
        self.calc_C()
        self.M = self.calc_M()

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, E, GS, backtrack_mat):
        """Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommended parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            E: np.ndarray (float32) of shape (h,w)
            GS: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a reference type. Changing it here may affect it on the outside.
        """
        raise NotImplementedError("TODO: Implement DPSeamImage.calc_bt_mat")
        h, w = M.shape


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    raise NotImplementedError("TODO: Implement scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    raise NotImplementedError("TODO: Implement resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)

    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org

    scaled_x_grid = [get_scaled_param(x, in_width, out_width) for x in range(out_width)]
    scaled_y_grid = [
        get_scaled_param(y, in_height, out_height) for y in range(out_height)
    ]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid, dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid, dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(
        image[y1s][:, x1s] * dx + (1 - dx) * image[y1s][:, x2s],
        (out_width, out_height, 3),
    )
    c2 = np.reshape(
        image[y2s][:, x1s] * dx + (1 - dx) * image[y2s][:, x2s],
        (out_width, out_height, 3),
    )
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(
        int
    )
    return new_image
