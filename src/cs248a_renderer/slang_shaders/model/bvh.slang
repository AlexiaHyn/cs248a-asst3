implementing "../model.slang";
import math;

public struct BVHNode
{
    public BoundingBox bound;
    public int left;
    public int right;
    public int primLeft;
    public int primRight;
    public int depth;

    public __init(BoundingBox bound, int left = -1, int right = -1, int primLeft = -1, int primRight = -1, int depth = -1)
    {
        this.bound = bound;
        this.left = left;
        this.right = right;
        this.primLeft = primLeft;
        this.primRight = primRight;
        this.depth = depth;
    }

    public float2 hit(Ray ray)
    {
        return this.bound.hit(ray);
    }
}

public struct BVH<T>
    where T : IRayCastable
{
    public struct HitResult
    {
        public T primitive;
        public RayHitResult rayHitResult;
    }

    // Slang buffer of BVHNode copied from Python.
    public RWStructuredBuffer<BVHNode> nodes;
    // Maximum number of nodes in the BVH, in this case, it'll be the size of the nodes buffer.
    public uint maxNodes;
    // Slang buffer of primitives represented by the BVH.
    public RWStructuredBuffer<T> primitives;
    // Number of primitives in the primitives buffer.
    public uint numPrimitives;

    public __init(RWStructuredBuffer<BVHNode> nodes, RWStructuredBuffer<T> primitives, uint numPrimitives, uint maxNodes)
    {
        this.nodes = nodes;
        this.primitives = primitives;
        this.numPrimitives = numPrimitives;
        this.maxNodes = maxNodes;
    }

    struct HitStack<int N>
    {
        uint[N] nodeIndices;
        uint top;

        __init()
        {
            this.top = 0;
        }

        [mutating]
        void push(uint nodeIndex)
        {
            this.nodeIndices[this.top] = nodeIndex;
            this.top += 1;
        }

        [mutating]
        uint pop()
        {
            this.top -= 1;
            return this.nodeIndices[this.top];
        }

        bool isEmpty()
        {
            return this.top == 0;
        }

        uint size()
        {
            return this.top;
        }
    }

    // BVH traversal to find the closest hit.
    public Optional<HitResult> hit(Ray ray)
    {
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        // TODO: Student implementation starts here.
        stack.push(0);
        float closestT = ray.tRange.y;

        while (!stack.isEmpty())
        {
            uint nodeIdx = stack.pop();
            if (nodeIdx >= this.maxNodes) continue;
            BVHNode node = this.nodes[nodeIdx];
            float2 tBox = node.hit(ray);
            if (tBox.x > tBox.y || tBox.y < ray.tRange.x || tBox.x > closestT)
            {
                continue;
            }

            bool isLeaf = (node.left < 0 && node.right < 0);
            if (isLeaf)
            {
                for (int i = node.primLeft; i < node.primRight; ++i)
                {
                    if (i < 0 || (uint)i >= this.numPrimitives) continue;
                    
                    Optional<RayHitResult> h = this.primitives[i].hit(ray);
                    
                    if (h.hasValue && h.value.t < closestT)
                    {
                        closestT = h.value.t;
                        HitResult hr;
                        hr.primitive = this.primitives[i];
                        hr.rayHitResult = h.value;
                        hitResult = Optional<HitResult>(hr);
                    }
                }
            }
            else
            {
                if (node.right >= 0) stack.push((uint)node.right);
                if (node.left >= 0)  stack.push((uint)node.left);
            }
        }

        // TODO: Student implementation ends here.

        return hitResult;
    }
}
