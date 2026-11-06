from dataclasses import dataclass, field

import numpy as np

from axial_perception.core.segments import fit_line


def canonical(d):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    return -d if d[2] < 0 else d


def perpendicular(v, d):
    return v - (v @ d) * d


@dataclass
class Landmark:
    class_id: int
    point: np.ndarray
    direction: np.ndarray
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    hits: int = 0


class LineLandmarks:
    """Stable ids for infinite lines re-observed in one fixed frame.

    A sighting matches the nearest same-class line within max_angle by perpendicular distance: below
    spawn_threshold it is EMA-merged (alpha), below min_anchor_separation it is matched without moving
    the anchor, otherwise it spawns a new id. Matched points are kept on a voxel grid.
    """

    def __init__(self, spawn_threshold, min_anchor_separation=None, alpha=0.0, max_angle=0.35, voxel=0.05):
        self.spawn_threshold = float(spawn_threshold)
        self.min_anchor_separation = float(spawn_threshold if min_anchor_separation is None
                                           else min_anchor_separation)
        if self.min_anchor_separation < self.spawn_threshold:
            raise ValueError(f"min_anchor_separation ({self.min_anchor_separation}) must be >= "
                             f"spawn_threshold ({self.spawn_threshold})")
        self.alpha = float(alpha)
        self.cos_max_angle = np.cos(max_angle)
        self.voxel = float(voxel)
        self.landmarks = {}
        self._next_id = 1

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
            lm.point = lm.point + self.alpha * perpendicular(c - lm.point, lm.direction)
            lm.direction = canonical(self.alpha * d + (1.0 - self.alpha) * lm.direction)
        elif gid is None or dist >= self.min_anchor_separation:
            gid = self._next_id
            self._next_id += 1
            self.landmarks[gid] = Landmark(int(class_id), c.copy(), d)
        lm = self.landmarks[gid]
        lm.hits += 1
        if pts is not None:
            lm.points = self.voxelize(np.concatenate([lm.points, pts]))
        return gid

    def update(self, xyz, cls, line):
        """Assigns every line of one frame (per-point line ids, 0 = none); returns line id -> landmark id."""
        ids = {}
        for k in np.unique(line[line > 0]):
            sel = line == k
            ids[int(k)] = self.assign(int(cls[sel][0]), *fit_line(xyz[sel]), xyz[sel])
        return ids

    def set_anchor(self, gid, point, direction, class_id=0):
        """Overwrites a line with an authoritative estimate; later ids never reuse gid."""
        gid = int(gid)
        lm = self.landmarks.get(gid)
        if lm is None:
            self.landmarks[gid] = Landmark(int(class_id), np.asarray(point, float).copy(), canonical(direction))
        else:
            lm.point, lm.direction = np.asarray(point, float).copy(), canonical(direction)
        self._next_id = max(self._next_id, gid + 1)

    def voxelize(self, pts):
        if self.voxel <= 0 or len(pts) == 0:
            return pts
        _, first = np.unique(np.floor(pts / self.voxel).astype(np.int64), axis=0, return_index=True)
        return pts[np.sort(first)]
