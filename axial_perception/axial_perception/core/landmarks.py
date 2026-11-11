from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from axial_perception.core.segments import fit_line


def canonical(d):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    return -d if d[2] < 0 else d


def perpendicular(v, d):
    return v - np.outer(v @ d, d) if v.ndim == 2 else v - (v @ d) * d


def rotation(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), a @ b
    if c < -1 + 1e-9:
        return np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1 + c)


@dataclass
class Landmark:
    class_id: int
    point: np.ndarray
    direction: np.ndarray
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    hits: int = 0


class LineLandmarks:
    """Stable ids for infinite lines re-observed in one fixed frame.

    Lines less elongated than min_elongation (first over second principal spread) have no reliable direction and are
    ignored. A sighting matches the nearest same-class line within max_angle by perpendicular distance: below
    spawn_threshold its points join the landmark and the line is refit to them, below min_anchor_separation it is
    matched without touching the landmark, otherwise it spawns a new id.
    """

    def __init__(self, spawn_threshold, min_anchor_separation=None, max_angle=0.35, voxel=0.05, min_elongation=0.0):
        self.spawn_threshold = float(spawn_threshold)
        self.min_anchor_separation = float(spawn_threshold if min_anchor_separation is None
                                           else min_anchor_separation)
        if self.min_anchor_separation < self.spawn_threshold:
            raise ValueError(f"min_anchor_separation ({self.min_anchor_separation}) must be >= "
                             f"spawn_threshold ({self.spawn_threshold})")
        self.cos_max_angle = np.cos(max_angle)
        self.voxel = float(voxel)
        self.min_elongation = float(min_elongation)
        self.landmarks = {}
        self._next_id = 1

    def accepts(self, pts, c):
        s = np.linalg.svd(pts - c, compute_uv=False)
        return len(s) >= 2 and s[0] >= self.min_elongation * s[1]

    def update(self, xyz, cls, line):
        """Assigns every accepted line of one frame (per-point line ids, 0 = none); returns line id -> landmark id."""
        ids = {}
        for k in np.unique(line[line > 0]):
            sel = line == k
            c, d = fit_line(xyz[sel])
            if self.accepts(xyz[sel], c):
                ids[int(k)] = self.assign(int(cls[sel][0]), c, d, xyz[sel])
        return ids

    def nearest(self, class_id, c, d):
        best, best_dist = None, np.inf
        for gid, lm in self.landmarks.items():
            if lm.class_id != class_id or abs(d @ lm.direction) < self.cos_max_angle:
                continue
            dist = np.linalg.norm(perpendicular(c - lm.point, lm.direction))
            if dist < best_dist:
                best, best_dist = gid, dist
        return best, best_dist

    def assign(self, class_id, c, d, pts=None):
        c, d = np.asarray(c, float), canonical(d)
        gid, dist = self.nearest(class_id, c, d)
        if gid is not None and dist < self.spawn_threshold:
            lm = self.landmarks[gid]
            if pts is not None:
                lm.points = self.voxelize(np.concatenate([lm.points, pts]))
                lm.point, direction = fit_line(lm.points)
                lm.direction = canonical(direction)
        elif gid is None or dist >= self.min_anchor_separation:
            gid = self._next_id
            self._next_id += 1
            self.landmarks[gid] = Landmark(int(class_id), c.copy(), d,
                                           self.voxelize(pts) if pts is not None else np.empty((0, 3)))
        self.landmarks[gid].hits += 1
        return gid

    def set_anchor(self, gid, point, direction, class_id=0):
        """Overwrites a line with an authoritative estimate and moves its points along, so later refits stay on it;
        later ids never reuse gid."""
        gid = int(gid)
        point, direction = np.asarray(point, float), canonical(direction)
        lm = self.landmarks.get(gid)
        if lm is None:
            self.landmarks[gid] = Landmark(int(class_id), point.copy(), direction)
        else:
            c = point + ((lm.point - point) @ direction) * direction
            lm.points = c + (lm.points - lm.point) @ rotation(lm.direction, direction).T
            lm.point, lm.direction = c, direction
        self._next_id = max(self._next_id, gid + 1)

    def snap(self, gid, point, direction):
        """set_anchor on an existing landmark, unless the estimate is beyond min_anchor_separation or max_angle from
        it: such an estimate is more likely another line than a correction of this one. Returns whether it snapped."""
        lm = self.landmarks.get(int(gid))
        if lm is None:
            return False
        point, direction = np.asarray(point, float), canonical(direction)
        if (abs(direction @ lm.direction) < self.cos_max_angle or
                np.linalg.norm(perpendicular(lm.point - point, direction)) > self.min_anchor_separation):
            return False
        self.set_anchor(gid, point, direction)
        return True

    def voxelize(self, pts):
        if self.voxel <= 0 or len(pts) == 0:
            return pts
        _, first = np.unique(np.floor(pts / self.voxel).astype(np.int64), axis=0, return_index=True)
        return pts[np.sort(first)]

    def dense_points(self, gid, min_neighbours):
        """Points of a landmark with at least min_neighbours others within 1.8 voxels (drops stray points)."""
        pts = self.landmarks[gid].points
        if min_neighbours <= 0 or len(pts) == 0:
            return pts
        n = cKDTree(pts).query_ball_point(pts, 1.8 * self.voxel, return_length=True) - 1
        return pts[n >= min_neighbours]
