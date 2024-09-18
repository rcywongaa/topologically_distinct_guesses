# Topo-Geometrically Distinct Path Computation

## Ideas
- Instead of doing 2 3D graphs, why not try 1 6D graph with coarser discretization?
- For finding distinct base trajectories, instead of constraining that the base is on the ground,
  perhaps we can consider constraining base below elbow (reasonable) to overcome infeasibility

## Requirements
- Given a vertex, return adjacent vertices
- (Given a cost) Determine the current wavefront
- Given the current wavefront and a vertex on the wavefront, determine adjacent vertices on the wavefront
- Given a set of adjacent vertices on a wavefront, determine the wavefront normal

## Problems
- Path hugging doesn't seem to work in positive curvature environments

## Neighborhood generation
Both methods will require keeping track of the current `NagNode` graph
### Vanilla
1. Generate subgraph from `visited`
   (note `visited` needs contain `NagNode`, not just `NodeIndex`)
1. Run A* or Dijkstra

### Use wavefront to skip generating subgraph?
(Requires uniformly expanding wavefront)
1. Check neighbors (taking into account nehgiborhood) of the current wavefront

## Notes
- The wavefront normal is less dependent on the path discretization than the path history
- Wavefront neighborhood
- The median curve / tween curve between the starting point and incident point
- We require two data structures
  - One priority queue / BinaryHeap of `NagNode`s tracking which cell to visit next
  - One HashSet of `NodeIndex` tracking the

## Mistakes
- Using `path_idx` to keep track of the path a node belongs to
  - `path_idx` is only determined later leading to complex lifetime/pointer/reference management
  - `path_idx` being mutable means it cannot be used as a hash to compare if nodes are "equal"
- Neighborhood is a set of `NagNode`s, not `NodeIndex` since the neighborhood of neighbors matter
- Prune nodes if neighborhood is subset of current neighbor
  - Never actually happens
- Distinct elbow trajectories don't really exist in many cases unless there are small floating obstacles
  Better to consider the whole robot instead of elbow and base separately
- Bulk processing multiple edges with similar cost doesn't work since the cost threshold changes depending on length
  ```
  fn pop_similar_cost(heap: &mut BinaryHeap<NagNode>) -> Option<Vec<NagNode>> {
    let popped = heap.pop()?;
    let popped_value = popped.value;
    let mut same_cost_pops = Vec::from_iter([popped]);
    /*
    Neither of these work.
    At longer distances, the threshold may need to be very large
    Unless we make heavy assumptions on the edge lengths
    */
    // let threshold = VALUE_DIFF_THRESHOLD;
    let threshold = *popped_value * VALUE_PROPORTION_THRESHOLD;
    while let Some(next_pop) = heap.peek() {
      if (popped_value - next_pop.value).abs() < threshold {
        same_cost_pops.push(heap.pop().unwrap());
      } else {
        break;
      }
    }
    Some(same_cost_pops)
  }
  ```

### Using nodes adjacent to parent as neighborhood doesn't work
- Depends on visit order
- Has no notion of the current wavefront
- Would always include the current node

## Design Decisions
- Used `three-d` instead of `kiss3d` due to the former having an easier API for finding picked geometry
- `ahash` used due to slow hashing of `std` implementation
- `NotNan1D` used because `Float` misses some sort traits

### Bin elbow circle vs Bin base circle
Bin elbow circle
Pros:
- Regular base positions
- Elbow circle should be smaller than base circle hence requiring less points
Cons:
- Not straightforward to match points on elbow circle to points on swept hexasphere
- Multiple base-eef positions may map to the same elbow position
- Binning may cause the violation of two constraints instead of one

Bin base circle
Pros:
- Regular elbow positions
- Checking visibility between base positions requires collision checking of 1 link as opposed to 2
Cons:
- Multiple elbow positions may map to the same base position

## Precondition
- Graph must be discretized in such a way that "adjacent" vertices are "similar" distance away

## Proof of Correctness
Show that this algorithm produces distinct paths iff the distinct paths cannot be deformed into each other without increasing the cost

Suppose the algorithm produces two paths A and B such that B can be deformed to A without increasing the cost

Suppose the algorithm produces one path A but there exists another path B which cannot be deformed into A without increasing the cost

- Neighborhoods grow at increasing cost

Example where this fails:
- Consider a 3x3 grid of points but the center and bottom right grid points very far from the top left
  With top left as start and bottom right as goal
  - This is only an issue because of our faulty implementation where we add children to nag not based on children cost but parent cost
- Consider a polar grid with 3 layers

Requires that the cost of a path increases monotonically with number of deformations from the shortest path

## Extensions
- Explore other mesh generation techniques for high dimension manifolds
  - What exactly is the distance metric for the elbow/base circles?
  - Delaunay triangulation works for higher dimension
    - Now the question is how to find uniformly distributed points on the 4D manifold
      - Fibonacci spiral for surface of spheres
      - Simulated annealing (https://math.stackexchange.com/questions/267859/optimal-distribution-of-points-in-a-cone)
- Notice that the relationship of the base sphere $r$ and the height of the elbow $h$
  expresses the surface of a sphere given by $l_1^2 = r^2 + h^2$
- Explain why non-uniform discretization leads to poor performance
- Including non-holonomic constraints / base heading in graph search problem
- Consider directed graphs (especially for eef path)
- Improving ease of use (e.g. incorporating it into moveit)
- More comparisons / benchmarking
- With more constraints (velocity constraints / acceleration constraints)
- 6 DoF arms
