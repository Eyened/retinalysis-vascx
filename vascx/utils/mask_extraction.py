import numpy as np
from scipy.linalg import lstsq
from scipy.ndimage import label, sobel
from skimage.morphology import binary_dilation, binary_erosion

from .transformation import get_affine_transform


class Bounds:
    def __init__(
        self, h, w, cy, cx, radius, min_x, min_y, max_x, max_y, *args, **kwargs
    ):
        self.h = h
        self.w = w
        self.cy = cy
        self.cx = cx
        self.radius = radius
        self.min_x = max(0, int(min_x))
        self.min_y = max(0, int(min_y))
        self.max_x = min(w, int(max_x))
        self.max_y = min(h, int(max_y))

    def init_coordinates(self):
        self.ys, self.xs = np.mgrid[: self.h, : self.w]

        self.xc = self.cx - self.xs
        self.yc = self.cy - self.ys
        self.r_squared = self.xc**2 + self.yc**2

    def make_binary_mask(self, mask_shrink_pixels=0):
        """
        creates a binary image of the bounds (circle and rectangle)
        """
        if not hasattr(self, "r_squared"):
            self.init_coordinates()

        d = mask_shrink_pixels
        r = self.radius - d

        mask = self.r_squared < r * r
        mask &= (self.xs > self.min_x + d) & (self.xs < self.max_x - d)
        mask &= (self.ys > self.min_y + d) & (self.ys < self.max_y - d)
        self.mask = mask
        return mask

    def background_mirroring(self, image, r_margin=4):
        """
        mirrors pixels around the box and circle defined by bounds
        Can be used in combination with contrast_enhance to avoid the bright boundary around the rim

        Parameters:
            image: image to mirror
            r_margin: margin in pixels to the bounds of the mask for doing the actual mirroring
        """
        result = np.copy(image)

        h, w = self.h, self.w
        min_y, max_y, min_x, max_x = self.min_y, self.max_y, self.min_x, self.max_x

        if not hasattr(self, "r_squared"):
            self.init_coordinates()

        r = np.sqrt(self.r_squared)
        # angle with respect to center (zero is top)
        th = np.arctan2(self.xc, self.yc)

        # d is the distance to the border of the disk for each pixel
        d = r - self.radius + r_margin
        # r2 is the distance of the mirrored point to the center
        r2 = self.radius - r_margin - d

        # y-coordinate of mirrored point
        y1 = np.array(np.round(self.cy - np.cos(th) * r2), int)
        y1 = np.clip(y1, 0, h - 1)

        # x-coordinate of mirrored point
        x1 = np.array(np.round(self.cx - np.sin(th) * r2), int)
        x1 = np.clip(x1, 0, w - 1)

        outside_circle = d >= 0

        # box mirroring
        min_y = min_y + r_margin
        max_y = max_y - r_margin
        min_x = min_x + r_margin
        max_x = max_x - r_margin
        result[:min_y, min_x:max_x] = result[2 * min_y : min_y : -1, min_x:max_x]
        result[max_y:, min_x:max_x] = result[max_y : 2 * max_y - h : -1, min_x:max_x]
        result[min_y:max_y, :min_x] = result[min_y:max_y, 2 * min_x : min_x : -1]
        result[min_y:max_y, max_x:] = result[min_y:max_y, max_x : 2 * max_x - w : -1]

        # disk mirroring
        result[outside_circle] = result[y1, x1][outside_circle]

        return result

    def get_cropping_matrix(self, target_diameter, patch_size=None):
        if patch_size is None:
            patch_size = target_diameter, target_diameter

        s = target_diameter / (2 * self.radius)
        scale = s, s
        center = self.cy, self.cx
        flip = False, False
        return get_affine_transform(patch_size, 0, scale, center, flip)

    def warp(self, transform, out_size):
        points = [
            [self.cx, self.cy],
            [self.min_x, self.min_y],
            [self.max_x, self.max_y],
        ]
        (cx, cy), (min_x, min_y), (max_x, max_y) = transform.apply(points)

        r0 = np.sqrt((self.min_x - self.cx) ** 2 + (self.min_y - self.cy) ** 2)
        r1 = np.sqrt((min_x - cx) ** 2 + (min_y - cy) ** 2)

        r = self.radius * (r1 / r0)
        h, w = out_size
        return Bounds(h, w, cy, cx, r, min_x, min_y, max_x, max_y)

    def to_dict(self):
        return {
            "disk": {
                "center": {"x": self.cx, "y": self.cy},
                "radius": self.radius,
            },
            "rectangle": {
                "min_x": self.min_x,
                "min_y": self.min_y,
                "max_x": self.max_x,
                "max_y": self.max_y,
            },
            "size": {"h": self.h, "w": self.w},
        }

    @staticmethod
    def from_dict(d):
        disk = d["disk"]
        rectangle = d["rectangle"]
        size = d["size"]
        return Bounds(
            h=size["h"],
            w=size["w"],
            cy=disk["center"]["y"],
            cx=disk["center"]["x"],
            radius=disk["radius"],
            min_x=rectangle["min_x"],
            min_y=rectangle["min_y"],
            max_x=rectangle["max_x"],
            max_y=rectangle["max_y"],
        )

    def _repr_markdown_(self):
        result = f"""
        #### Bounds:

        - Center: ({self.cx}, {self.cy})
        - Radius: {self.radius}
        - Rectangle: ({self.min_x}, {self.min_y}) - ({self.max_x}, {self.max_y})
        """
        return result.strip()


def mode(array, res=128):
    d = len(array) / res
    c = d * np.bincount(np.round(array / d).astype(int)).argmax()
    window = array[np.abs(array - c) < d]
    return np.bincount(np.round(window).astype(int)).argmax()


def circle_fit(pts):
    # adapted from: https://scikit-guess.readthedocs.io/en/latest/generated/skg.nsphere.html
    B = np.ones((len(pts), 3))
    B[:, :-1] = pts
    d = (pts**2).sum(axis=-1)
    y, *_ = lstsq(B, d, overwrite_a=True, overwrite_b=True)
    c = 0.5 * y[:2]
    r = np.sqrt(y[-1] + (c**2).sum())
    return r, c


def ransac_circle_fit(
    pts, min_r, max_r, num_iterations=1000, n_points=5, inlier_threshold=2
):
    best_inliers = None

    for _ in range(num_iterations):
        while True:
            random_indices = np.random.choice(len(pts), n_points, replace=False)
            sampled_points = pts[random_indices]

            r, c = circle_fit(sampled_points)
            if min_r < r and r < max_r:
                break

        distances = np.abs(np.sqrt(np.sum((pts - c) ** 2, axis=-1)) - r)
        inliers = distances < inlier_threshold

        if best_inliers is None or np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
    return circle_fit(pts[best_inliers]), len(best_inliers)


def has_mask(im, r):
    h, w = im.shape
    corner_mask = np.ones((h, w), dtype=bool)
    center_mask = np.ones((h, w), dtype=bool)

    cy, cx = h // 2, w // 2
    r_fit = max(cy, cx)

    corner_mask = np.ones((h, w), dtype=bool)
    center_mask = np.ones((h, w), dtype=bool)

    corner_mask[r < r_fit] = 0
    center_mask[r > 0.25 * r_fit] = 0

    corner = np.mean(im[corner_mask])
    center = np.mean(im[center_mask])
    return corner < 0.2 * center


def find_image_edge(im, dy, dx, r, min_r, max_r, include_prior=True):
    h, w = im.shape

    footprint = [(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)]

    im_region_small = im > 0.2
    im_region_shrunk = binary_erosion(im_region_small, footprint=footprint)

    im_region_large = im > 0.05
    labels, _ = label(im_region_large)
    i = np.argmax(np.bincount(labels.flat)[1:]) + 1
    im_region_grown = binary_dilation(labels == i, footprint=footprint)

    mask = (r > min_r) & (r < max_r) & im_region_grown & ~im_region_shrunk

    # expected direction of gradient
    dy_norm, dx_norm = dy[mask] / r[mask], dx[mask] / r[mask]
    # gradient
    gy, gx = sobel(im, 0), sobel(im, 1)

    max_edge = np.zeros((h, w))
    max_edge[mask] = gx[mask] * dx_norm + gy[mask] * dy_norm

    if include_prior:
        r_est = 0.5 * (min_r + max_r)
        # multiply with prior (p = 0.1 at center)
        a = np.log(0.1) / r_est**2
        prior = np.exp(a * (r - r_est) ** 2)

        max_edge[mask] *= prior[mask]

    return max_edge


def find_edges(max_edge, cx, cy, min_val):
    left = np.argmax(max_edge[:, :cx], axis=1)
    left_mask = np.max(max_edge[:, :cx], axis=1) > min_val

    right = cx + np.argmax(max_edge[:, cx:], axis=1)
    right_mask = np.max(max_edge[:, cx:], axis=1) > min_val

    top = np.argmax(max_edge[:cy], axis=0)
    top_mask = np.max(max_edge[:cy], axis=0) > min_val

    bottom = cy + np.argmax(max_edge[cy:], axis=0)
    bottom_mask = np.max(max_edge[cy:], axis=0) > min_val

    edge_points = left, right, top, bottom
    edge_masks = left_mask, right_mask, top_mask, bottom_mask
    return edge_points, edge_masks


def extract_bounds(image, include_prior=True, min_val=0.1):
    if len(image.shape) == 3:
        im = image[:, :, 0]  # red channel
    elif len(image.shape) == 2:
        im = image

    h, w = im.shape
    cy, cx = h // 2, w // 2

    # conservative estimates
    min_r = min(h, w) / 6
    max_r = 0.6 * max(h, w)

    ys, xs = np.mgrid[:h, :w]
    dy = cy - ys
    dx = cx - xs
    r = np.sqrt(dy**2 + dx**2)

    if not has_mask(im, r):
        radius = np.sqrt(cx**2 + cy**2)
        min_x, max_x = 0, w
        min_y, max_y = 0, h
        return Bounds(h, w, cy, cx, radius, min_x, min_y, max_x, max_y)

    # setting border to 0
    # this is needed to find the box edge the circle mask is not contained in the image
    # TODO: check!
    im[0] = im[-1] = im[:, 0] = im[:, -1] = 0

    # filtered image with high response at edge of fundus mask
    max_edge = find_image_edge(im, dy, dx, r, min_r, max_r, include_prior)

    # points on left, right, top bottom of the mask (arrays 0-w or 0-h)
    # masks for valid points
    edge_points, edge_masks = find_edges(max_edge, cx, cy, min_val)

    # edges of box at most occuring value
    box = [mode(pts[m]) for pts, m in zip(edge_points, edge_masks)]

    d = 0.005 * min(h, w)
    rect_masks = [np.abs(val - pts) < d for pts, val in zip(edge_points, box)]

    # assign rect bounds if it has more than 20% support
    has_rect_edge = [np.mean(r) > 0.2 for r in rect_masks]

    rect = [
        box_edge if r else img_bounds
        for img_bounds, box_edge, r in zip((0, w, 0, h), box, has_rect_edge)
    ]

    combined_masks = [
        edge_mask & ~rect_mask for edge_mask, rect_mask in zip(edge_masks, rect_masks)
    ]

    left, right, top, bottom = edge_points

    x_pts = [left, right, np.arange(w), np.arange(w)]
    y_pts = [np.arange(h), np.arange(h), top, bottom]
    circle_pts = np.concatenate(
        [[x[m], y[m]] for x, y, m in zip(x_pts, y_pts, combined_masks)], axis=-1
    ).T

    (radius, (cx, cy)), n_pts_ransac = ransac_circle_fit(circle_pts, min_r, max_r)
    min_x, max_x, min_y, max_y = rect

    # this is a hack for certain type of images in range ~490000
    if (h, w) == (1372, 1359) and n_pts_ransac < 1200 and np.sum(has_rect_edge) == 4:
        min_x, max_x, min_y, max_y = rect
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        radius = 0.67 * min((max_x - min_x), (max_y - min_y))

        return Bounds(h, w, cy, cx, radius, min_x, min_y, max_x, max_y)

    return Bounds(h, w, cy, cx, radius, min_x, min_y, max_x, max_y)
