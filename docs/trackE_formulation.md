# Track E: DabbaDash — Optimizing Last-Mile Delivery in Mumbai

## Problem Statement

DabbaDash models last-mile delivery operations across the metropolitan expanse of Mumbai as a Capacitated Vehicle Routing Problem (CVRP). The distribution network consists of:
- A central distribution depot located in Andheri East (coordinates: $19.1136^\circ\text{ N}, 72.8697^\circ\text{ E}$).
- A set of $N$ geographically dispersed customer delivery points across Greater Mumbai.
- A homogeneous fleet of delivery vehicles, each with an identical maximum load capacity $Q$.
- An unconstrained fleet size (vehicles can be dispatched as needed from the depot).

The primary goal is to find a set of vehicle routes that services every customer's demand while respecting vehicle capacity limits, starting and finishing each tour at the central depot, and minimizing the total distance traveled across all routes.

---

## Mathematical Formulation

### Sets and Indices
- $V = \{0, 1, 2, \dots, N\}$: Set of all nodes, where $0$ denotes the central depot in Andheri East.
- $V_c = \{1, 2, \dots, N\}$: Set of customer delivery locations ($V = V_c \cup \{0\}$).
- $A = \{(i, j) : i, j \in V, i \neq j\}$: Set of directed arcs connecting nodes.

### Parameters
- $d_{ij} \ge 0$: Travel distance (in kilometers) between node $i$ and node $j$, computed as great-circle haversine distance scaled by a circuity factor of $1.3$. By definition, $d_{ii} = 0$.
- $q_i \in \{1, \dots, 9\}$: Delivery demand of customer $i \in V_c$. For the depot, $q_0 = 0$.
- $Q = 50$: Maximum payload capacity of each homogeneous delivery vehicle.

### Decision Variables
- $x_{ij} \in \{0, 1\}$: Binary decision variable equal to $1$ if a vehicle travels directly along arc $(i, j) \in A$, and $0$ otherwise.
- $u_i \in \mathbb{R}$: Continuous auxiliary variable representing the cumulative vehicle load after servicing node $i \in V_c$ (used for MTZ-style capacity tracking and subtour elimination).

### Objective Function
Minimize total travel distance across all delivery tours:

$$\min \sum_{i \in V} \sum_{j \in V, j \neq i} d_{ij} x_{ij}$$

### Constraints

1. **Customer Visit Requirements:**
   Each customer is visited and departed from exactly once:
   $$\sum_{j \in V, j \neq i} x_{ij} = 1 \quad \forall i \in V_c$$
   $$\sum_{i \in V, i \neq j} x_{ij} = 1 \quad \forall j \in V_c$$

2. **Depot Flow Conservation & Vehicle Balance:**
   The number of vehicles departing from the depot equals the number of vehicles returning:
   $$\sum_{j \in V_c} x_{0j} = \sum_{i \in V_c} x_{i0} = K$$
   where $K \ge 1$ is the total number of vehicles dispatched.

3. **Subtour Elimination and Capacity Constraints (MTZ formulation):**
   To prevent detached sub-cycles that do not connect to the depot and to enforce that the accumulated route load never exceeds vehicle capacity $Q$:
   $$u_i - u_j + Q x_{ij} \le Q - q_j \quad \forall i, j \in V_c, i \neq j$$
   $$q_i \le u_i \le Q \quad \forall i \in V_c$$

4. **Integrality and Domain Constraints:**
   $$x_{ij} \in \{0, 1\} \quad \forall (i, j) \in A$$
   $$u_i \ge 0 \quad \forall i \in V_c$$

---

## Assumptions

1. **Synthetic Spatial Distribution:** Customer locations are generated via Gaussian mixtures centered around key commercial and residential hubs in Mumbai (Andheri, Bandra, Powai, Borivali, Dadar, Ghatkopar).
2. **Circuity Adjustment:** Ground road distances in Mumbai are approximated by multiplying great-circle Haversine distances by an empirical circuity factor of $1.3$, accounting for road curvature, coastal geography, and highway networks.
3. **Distance Metric:** Constant vehicle travel speed is assumed; total route distance (km) serves as the primary optimization metric rather than dynamic travel time.
4. **Demand Distribution:** Each customer demands an integer quantity $q_i \in [1, 9]$ items/units.
5. **Vehicle Capacity:** Vehicle capacity is fixed at $Q = 50$ units per vehicle.
6. **Time Windows:** No delivery time windows or service durations are considered in the baseline formulation.

---

## Evaluation Metrics

- **Total Distance (km):** The sum of arc travel distances over all completed vehicle routes.
- **Number of Vehicles:** The total count of active delivery routes dispatched from and returned to the depot.
- **Runtime (s):** Wall-clock execution time required by the routing heuristic or solver, measured using high-precision timers (`time.perf_counter`).
- **Feasibility:** Independent verification that:
  - Every customer is visited exactly once.
  - Every route starts and terminates at depot node $0$.
  - Cumulative demand per route does not exceed capacity $Q = 50$.

---

## Baseline Heuristics

1. **Driver Order Heuristic:**
   - Visits customers sequentially in index order ($1, 2, \dots, N$).
   - Appends customers to the current route while cumulative demand $\le Q$.
   - When the next customer would cause the vehicle load to exceed $Q$, the vehicle returns to the depot ($0$), and a new vehicle departs with that customer.

2. **Nearest-Neighbour Heuristic (Greedy with Capacity):**
   - Dispatches a vehicle from depot $0$.
   - At each step, evaluates all remaining unvisited customers whose demand fits within the remaining vehicle capacity ($Q - \text{current load}$).
   - Steps to the closest eligible customer (minimizing $d_{\text{current}, j}$).
   - If no unvisited customer fits in the remaining capacity, the vehicle returns to depot $0$, and a fresh vehicle is dispatched.

---

## Limitations

- **Synthetic Customer Data:** The customer coordinates and demand figures are synthetically generated rather than sampled from historical Mumbai delivery logs.
- **Traffic and Congestion:** Mumbai's dynamic diurnal traffic variations, toll plazas (e.g., Sea Link), and one-way bottlenecks are not modeled; routing relies strictly on distance with a static circuity factor.
- **Road Network Topology:** Actual turn restrictions, sea bridge constraints, and non-Euclidean connectivity (e.g., east-west railway crossing bottlenecks) are approximated via the circuity factor rather than a full road-network graph.
