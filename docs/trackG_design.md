# Track G: Operating Systems Optimization (xv6 Lottery Scheduler) — Design Document

## 1. Operating System Concepts

- **Process Scheduling:** The operating system scheduler allocates CPU execution time to competing runnable threads. Modern kernels balance throughput, latency, and fairness across diverse workloads.
- **Run Queue Scan:** xv6 uses a simple tabular process list `proc[NPROC]` rather than a complex multi-level feedback queue. In the default round-robin scheduler, the CPU scans the array linearly, picking the first `RUNNABLE` process, running it for a quantum (until a timer interrupt yields), and continuing sequentially.
- **Lottery Scheduling:** A randomized, proportional-share scheduling algorithm where processes are assigned lottery tickets. In each scheduling decision, the scheduler tallies the tickets of all currently `RUNNABLE` processes ($T = \sum_i t_i$), draws a winning ticket uniformly at random ($w \in [0, T)$), and runs the process whose ticket interval contains $w$. Each runnable process wins with probability $t_i / T$.
- **Context Switch via `swtch`:** CPU core context switching is implemented in assembly (`kernel/swtch.S`). It saves the callee-saved registers (`ra`, `sp`, `s0`–`s11`) of the old thread context and restores those of the incoming thread (either scheduler context `c->context` or process context `p->context`).
- **Synchronization via `p->lock` Spinlocks:** Each process table entry is protected by its own spinlock `p->lock`. The scheduler acquires `p->lock` before inspecting `p->state`, sets `RUNNING`, switches context while holding the lock, and releases the lock only after the process returns control or yields. In lottery scheduling, holding two process locks simultaneously is strictly avoided to prevent deadlock across multiple CPUs.

---

## 2. Experimental Objectives

1. **Proportional-Share Allocation:** Verify whether lottery scheduling achieves CPU time allocation proportional to ticket endowments ($30:20:10 = 50\% : 33.3\% : 16.7\%$), contrasting with round-robin which enforces uniform $1/3$ shares regardless of tickets.
2. **Turnaround Time Latency:** Measure the turnaround time of a short interactive-like job competing against long-running CPU hogs, and quantify the latency reduction achieved by boosting its ticket priority.
3. **Short-Term Fairness & Variance:** Quantify the variance of CPU shares among equal-priority processes using Jain's Fairness Index and the Coefficient of Variation ($CV$). While round-robin provides deterministic short-term fairness, lottery scheduling trades short-term deterministic fairness for stateless probabilistic proportional-share allocation.

---

## 3. Measured Time Resolution (xv6 Ticks)

As inspected directly in the kernel source:
- In `kernel/start.c` (`timerinit`) and `kernel/trap.c` (`clockintr`):
  ```c
  w_stimecmp(r_time() + 1000000);
  ```
- In QEMU's RISC-V virt machine platform, the hardware `mtime` clock frequency is 10 MHz ($10^7$ cycles/sec).
- 1,000,000 cycles corresponds to exactly:
  $$\frac{1,000,000}{10,000,000\text{ cycles/sec}} = 0.1\text{ seconds} = 100\text{ ms}$$
- Therefore, one xv6 timer tick is **100 ms (10 Hz)**, and the time resolution of `uptime()` is **1 tick = 100 ms**.

---

## 4. Benchmark Workloads (`user/schedbench.c`)

All measurements use a tight volatile-counter spin loop. Syscall overhead is amortized by sampling `uptime()` only once every 10,000 loop iterations.

- **Calibration (Baseline):**
  A single process spins alone on the CPU for 20 ticks to determine the single-core iteration rate (`iters_per_tick`). This baseline sets the work quantum for calibrated short jobs.
- **Workload 1 (Proportional Share):**
  Three CPU-bound child processes are forked with ticket allocations of 30, 20, and 10. All three spin concurrently until a shared deadline of 60 ticks from the common start. Iteration counts are reported to the parent via a pipe to determine observed work shares.
- **Workload 2 (Turnaround Time):**
  Four CPU-bound "hog" processes (10 tickets each) spin for 100 ticks. Concurrently, one short job executes a fixed work budget of $15 \times \text{iters\_per\_tick}$. Two variants are compared:
  - Variant (a): Short job has 10 tickets (equal priority with hogs).
  - Variant (b): Short job has 40 tickets (boosted priority).
  Turnaround time (ticks to completion from start) is recorded.
- **Workload 3 (Short-Term Fairness):**
  Eight processes with equal tickets (10 each) spin concurrently until a shared deadline of 60 ticks from start. Work distributions are evaluated using:
  - Jain's Fairness Index: $J = \frac{(\sum x_i)^2}{n \sum x_i^2} \in [1/n, 1.0]$
  - Coefficient of Variation: $CV = \sigma / \mu$

---

## 5. Single-CPU Configuration (`CPUS=1`)

The benchmark harness explicitly sets `CPUS=1` (`make qemu CPUS=1`):
- In a multi-core environment (`CPUS > 1`), multiple runnable threads execute concurrently across distinct hardware cores, masking scheduling imbalances and introducing inter-core cache, memory bus, and lock contention.
- Restricting execution to a single core ensures that all runnable processes directly compete for the same execution pipeline and time slices, providing an isolated, clean testbed for scheduler policy evaluation.

---

## 6. Experimental Limitations

1. **Fixed-Seed PRNG:** The kernel lottery scheduler uses a 32-bit xorshift generator with a fixed non-zero seed (`2463534242UL`). While this guarantees determinism and reproducibility across benchmark runs, real-world lottery schedulers rely on high-entropy random sources to prevent adversarial starvation.
2. **Tick Coarseness:** With an `uptime()` resolution of 100 ms, sub-tick durations cannot be directly observed from user space.
3. **QEMU Virtualized Timing:** QEMU executes in software emulation; CPU instruction dispatch speed varies with the host machine's architecture, CPU frequency scaling, and background OS load.
4. **Host Load Dependence:** Variations in host scheduling latency can introduce timing jitter into the emulated RISC-V timebase register.
5. **Sample Size:** Each workload is repeated 5 times. While sufficient to capture central tendencies and standard deviations, statistical confidence intervals remain bounded by the sample count.
