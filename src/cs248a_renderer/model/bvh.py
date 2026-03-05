import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import numpy as np
import slangpy as spy

from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.primitive import Primitive
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class BVHNode:
    # The bounding box of this node.
    bound: BoundingBox3D = field(default_factory=BoundingBox3D)
    # The index of the left child node, or -1 if this is a leaf node.
    left: int = -1
    # The index of the right child node, or -1 if this is a leaf node.
    right: int = -1
    # The starting index of the primitives in the primitives array.
    prim_left: int = 0
    # The ending index (exclusive) of the primitives in the primitives array.
    prim_right: int = 0
    # The depth of this node in the BVH tree.
    depth: int = 0

    def get_this(self) -> Dict:
        return {
            "bound": self.bound.get_this(),
            "left": self.left,
            "right": self.right,
            "primLeft": self.prim_left,
            "primRight": self.prim_right,
            "depth": self.depth,
        }

    @property
    def is_leaf(self) -> bool:
        """Checks if this node is a leaf node."""
        return self.left == -1 and self.right == -1


class BVH:
    def __init__(
        self,
        primitives: List[Primitive],
        max_nodes: int,
        min_prim_per_node: int = 1,
        num_thresholds: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Builds the BVH from the given list of primitives. The build algorithm should
        reorder the primitives in-place to align with the BVH node structure.
        The algorithm will start from the root node and recursively partition the primitives
        into child nodes until the maximum number of nodes is reached or the primitives
        cannot be further subdivided.
        At each node, the splitting axis and threshold should be chosen using the Surface Area Heuristic (SAH)
        to minimize the expected cost of traversing the BVH during ray intersection tests.

        :param primitives: the list of primitives to build the BVH from
        :type primitives: List[Primitive]
        :param max_nodes: the maximum number of nodes in the BVH
        :type max_nodes: int
        :param min_prim_per_node: the minimum number of primitives per leaf node
        :type min_prim_per_node: int
        :param num_thresholds: the number of thresholds per axis to consider when splitting
        :type num_thresholds: int
        """
        self.nodes: List[BVHNode] = []

        # TODO: Student implementation starts here.
        self.primitives = primitives
        self.max_nodes = max_nodes
        self.min_prim_per_node = min_prim_per_node
        self.num_thresholds = num_thresholds
        self.on_progress = on_progress

        # Initialize the root node
        root = BVHNode(
            prim_left=0,
            prim_right=len(primitives),
            depth=0
        )
        self.nodes.append(root)
        
        # Build recursively
        self._build(0)

    def _build(self, root_idx: int = 0) -> None:
        queue = [root_idx]
        qptr = 0

        while qptr < len(queue):
            node_idx = queue[qptr]
            qptr += 1

            node = self.nodes[node_idx]
            prim_left, prim_right = node.prim_left, node.prim_right
            num_prims = prim_right - prim_left
            if num_prims <= 0:
                continue
            
            # calcuate bounds
            node_bound = self.primitives[prim_left].bounding_box
            centers = []

            for i in range(prim_left, prim_right):
                bb = self.primitives[i].bounding_box
                node_bound = BoundingBox3D.union(node_bound, bb)
                c = bb.center
                centers.append([c.x, c.y, c.z])

            node.bound = node_bound
            centers = np.asarray(centers, dtype=np.float32)

            # stop splitting if number of primitives in a node reaches min_prim_per_nodes
            if num_prims <= self.min_prim_per_node:
                continue
            # stop splitting if total number of nodes reaches max_nodes
            if len(self.nodes) + 2 > self.max_nodes:
                continue

            # caculate SAH cost and find the right SAH split
            cmin = centers.min(axis=0)
            cmax = centers.max(axis=0)

            best_cost = float("inf")
            best_axis = -1
            best_thr = 0.0

            for axis in range(3):
                extent = float(cmax[axis] - cmin[axis])
                if extent < 1e-6:
                    continue
                # make partitions
                for t in range(1, self.num_thresholds):
                    thr = float(cmin[axis] + extent * (t / self.num_thresholds))

                    l_bb = BoundingBox3D()
                    r_bb = BoundingBox3D()
                    l_cnt = 0
                    r_cnt = 0

                    for i in range(prim_left, prim_right):
                        bb = self.primitives[i].bounding_box
                        if bb.center[axis] < thr:
                            l_bb = BoundingBox3D.union(l_bb, bb)
                            l_cnt += 1
                        else:
                            r_bb = BoundingBox3D.union(r_bb, bb)
                            r_cnt += 1

                    if l_cnt == 0 or r_cnt == 0:
                        continue

                    cost = l_cnt * l_bb.area + r_cnt * r_bb.area
                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_thr = thr
            # if SAH split not found
            if best_axis == -1:
                ext = cmax - cmin
                best_axis = int(np.argmax(ext))
                if float(ext[best_axis]) < 1e-4:
                    continue
                best_thr = float(np.median(centers[:, best_axis]))

            # Make the split according to best threshold found
            mid = self._partition(prim_left, prim_right, best_axis, best_thr)

            if mid == prim_left or mid == prim_right:
                self.primitives[prim_left:prim_right] = sorted(
                    self.primitives[prim_left:prim_right],
                    key=lambda p: p.bounding_box.center[best_axis],
                )
                mid = (prim_left + prim_right) // 2
                if mid == prim_left or mid == prim_right:
                    continue

            left_idx = len(self.nodes)
            right_idx = left_idx + 1

            self.nodes.append(BVHNode(prim_left=prim_left, prim_right=mid, depth=node.depth + 1))
            self.nodes.append(BVHNode(prim_left=mid, prim_right=prim_right, depth=node.depth + 1))

            node.left = left_idx
            node.right = right_idx

            if self.on_progress:
                self.on_progress(len(self.nodes), self.max_nodes)

            # Enqueue children so both sides get split fairly
            queue.append(left_idx)
            queue.append(right_idx)

    def _partition(self, left: int, right: int, axis: int, threshold: float) -> int:
        i = left
        for j in range(left, right):
            if self.primitives[j].bounding_box.center[axis] < threshold:
                self.primitives[i], self.primitives[j] = self.primitives[j], self.primitives[i]
                i += 1
        return i
    
        # TODO: Student implementation ends here.


def create_bvh_node_buf(module: spy.Module, bvh_nodes: List[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
