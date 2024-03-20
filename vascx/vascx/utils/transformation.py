import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from sklearn.preprocessing import PolynomialFeatures


class ProjectiveTransform:
    def __init__(self, M):
        self.M = M
        self.M_inv = np.linalg.inv(M)

    def apply(self, points):
        # Add homogeneous coordinate (1) to each point
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        p = np.dot(points_homogeneous, self.M.T)
        # Normalize by dividing by the last column (homogeneous coordinate)
        return p[:, :2] / p[:, [-1]]

    def apply_inverse(self, points):
        # Add homogeneous coordinate (1) to each point
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        p = np.dot(points_homogeneous, self.M_inv.T)
        # Normalize by dividing by the last column (homogeneous coordinate)
        return p[:, :2] / p[:, [-1]]

    def get_dsize(self, image, out_size):
        if out_size is None:
            h, w = image.shape[:2]
            corners = np.array([[0, 0], [0, h], [w, h], [w, 0]])
            return np.ceil(self.apply(corners).max(axis=0)).astype(int)
        else:
            h, w = out_size
            return w, h

    def warp(self, image, out_size=None):
        dsize = self.get_dsize(image, out_size)

        if image.dtype == bool or image.dtype == np.uint8:
            warped = cv2.warpPerspective(
                image, self.M, dsize=dsize, flags=cv2.INTER_NEAREST
            )
        else:
            warped = cv2.warpPerspective(image, self.M, dsize=dsize)
        return warped

    def warp_inverse(self, image, out_size=None):
        dsize = self.get_dsize(image, out_size)

        if image.dtype == bool or image.dtype == np.uint8:
            warped = cv2.warpPerspective(
                image, self.M_inv, dsize=dsize, flags=cv2.INTER_NEAREST
            )
        else:
            warped = cv2.warpPerspective(image, self.M_inv, dsize=dsize)
        return warped

    def _repr_html_(self):
        html_table = "<h4>Projective Transform:</h4><table>"

        for row in self.M:
            html_table += "<tr>"
            for val in row:
                html_table += f"<td>{val:.3f}</td>"
            html_table += "</tr>"

        html_table += "</table>"
        return html_table


class ParabolicTransform:
    def __init__(self, model_dx, model_dy):
        self.model_dx = model_dx
        self.model_dy = model_dy

    def apply(self, points):
        points_poly = PolynomialFeatures(degree=2).fit_transform(points)
        dx = self.model_dx.predict(points_poly)
        dy = self.model_dy.predict(points_poly)
        return np.array(points) + np.array([dx, dy]).T

    def apply_inverse(self, points):
        points_poly = PolynomialFeatures(degree=2).fit_transform(points)
        dx = self.model_dx.predict(points_poly)
        dy = self.model_dy.predict(points_poly)
        return np.array(points) - np.array([dx, dy]).T

    def warp(self, image, out_size, fraction=1.0):
        h, w = out_size
        ys, xs = np.mgrid[0:h, 0:w]
        all_pixels = np.array([xs.flatten(), ys.flatten()]).T
        pixels_parabola = PolynomialFeatures(degree=2).fit_transform(all_pixels)
        dx = self.model_dx.predict(pixels_parabola).reshape(h, w)
        dy = self.model_dy.predict(pixels_parabola).reshape(h, w)
        pixels_mapped = np.array(
            [(ys + fraction * dy).flatten(), (xs + fraction * dx).flatten()]
        )

        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = map_coordinates(
                    image[:, :, i], pixels_mapped
                ).reshape(h, w)
        else:
            result = map_coordinates(image, pixels_mapped).reshape(h, w)
        return result

    def warp_inverse(self, image, out_size):
        return self.warp(image, out_size, fraction=-1.0)

    def _repr_markdown_(self):
        def _coeficents(model):
            return ", ".join(f"{x:.3f}" for x in [model.intercept_, *model.coef_])

        result = f"""
        #### ParabolicTransform:
        - dx: ({_coeficents(self.model_dx)}
        - dy: ({_coeficents(self.model_dy)}
        """
        return result.strip()


class CombinedTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def apply(self, points):
        result = points
        for transform in self.transforms:
            result = transform.apply(result)
        return result

    def warp(self, image, out_size):
        h, w = out_size
        ys, xs = np.mgrid[0:h, 0:w]
        all_pixels = np.array([xs.flatten(), ys.flatten()]).T
        mapped_pixels = all_pixels
        for transform in self.transforms[::-1]:
            mapped_pixels = transform.apply_inverse(mapped_pixels)
        xs, ys = mapped_pixels.T

        if len(image.shape) == 3:
            result = np.zeros((h, w, 3))
            for i in range(3):
                result[:, :, i] = map_coordinates(image[:, :, i], (ys, xs)).reshape(
                    h, w
                )
        else:
            result = map_coordinates(image, (ys, xs)).reshape(h, w)

        return result

    def warp_sequence(self, image, out_size):
        result = image
        for transform in self.transforms:
            result = transform.warp(result, out_size)
        return result


def get_affine_transform(out_size, rotate, scale, center, flip):
    """
    Parameters:
    out_size: size of the extracted patch (h, w)
    rotate: angle in degrees
    scale: scaling factor (sy, sx)
    center: center of the patch (cy, cx)
    flip: apply horizontal/vertical flipping
    """
    # center to top left corner
    cy, cx = center
    C1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)

    # rotate
    th = rotate * np.pi / 180
    R = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]],
        dtype=float,
    )

    # scale
    sy, sx = scale
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=float)

    # top left corner to center
    h, w = out_size
    ty = h / 2
    tx = w / 2
    C2 = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=float)

    M = C2 @ S @ R @ C1
    flip_vertical, flip_horizontal = flip

    if flip_horizontal:
        M = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]]) @ M
    if flip_vertical:
        M = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]]) @ M

    return ProjectiveTransform(M)


def approximate_affine_transform(source_points, target_points):
    n = min(len(source_points), len(target_points))

    if n < 3:
        # Handle case with fewer than 3 point pairs
        raise ValueError("Insufficient point pairs to approximate affine transform")

    A = np.array(
        [[xs, ys, 1, 0, 0, 0] for (xs, ys) in source_points]
        + [[0, 0, 0, xs, ys, 1] for (xs, ys) in source_points]
    )

    B = np.array(
        [xt for (xt, yt) in target_points] + [yt for (xt, yt) in target_points]
    )

    coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    matrix = np.reshape(np.append(coefficients, [0, 0, 1]), (3, 3))
    return ProjectiveTransform(matrix)


def approximate_affine_coefficients(source_points, target_points):
    n = min(len(source_points), len(target_points))

    if n < 3:
        # Handle case with fewer than 3 point pairs
        raise ValueError("Insufficient point pairs to approximate affine transform")

    A = np.array(
        [[xs, ys, 1, 0, 0, 0] for (xs, ys) in source_points]
        + [[0, 0, 0, xs, ys, 1] for (xs, ys) in source_points]
    )

    B = np.array(
        [xt for (xt, yt) in target_points] + [yt for (xt, yt) in target_points]
    )

    coefficients, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    return coefficients


def exact_affine_coefficients(source_points, target_points):
    # source_points and target_points should be exactly 3 each

    A = np.vstack([source_points.T, np.ones((1, 3))])
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("Points are collinear. Unable to compute an affine transform.")

    return np.dot(target_points.T, A_inv).flatten()


def apply_coefficients(coefficients, points):
    c = coefficients
    xs = c[0] * points[:, 0] + c[1] * points[:, 1] + c[2]
    ys = c[3] * points[:, 0] + c[4] * points[:, 1] + c[5]

    return np.array([xs, ys]).T


def approximate_projective_transform(source_points, target_points):
    # based on: https://github.com/opencv/opencv/blob/11b020b9f9e111bddd40bffe3b1759aa02d966f0/modules/imgproc/src/imgwarp.cpp#L3001
    n = min(len(source_points), len(target_points))
    if n < 4:
        raise ValueError(
            "Insufficient point pairs to approximate a projective transform"
        )

    A = []
    B = []
    for (xs, ys), (xt, yt) in zip(source_points, target_points):
        A.append((xs, ys, 1, 0, 0, 0, -xs * xt, -ys * xt))
        B.append(xt)
        A.append((0, 0, 0, xs, ys, 1, -xs * yt, -ys * yt))
        B.append(yt)

    a, c, e, b, d, f, g, h = np.linalg.pinv(A) @ B
    return ProjectiveTransform(np.array([[a, c, e], [b, d, f], [g, h, 1]]))
