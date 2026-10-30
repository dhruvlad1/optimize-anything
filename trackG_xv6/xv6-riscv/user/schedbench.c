#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

// Communication packet sent from each child worker to the parent via pipe.
struct result_msg {
  int id;
  uint64 work;
};

// Helper function to print 64-bit unsigned integers in decimal without relying on %lu.
static void
print_u64(uint64 n)
{
  char buf[24];
  int i = 0;
  if (n == 0) {
    printf("0");
    return;
  }
  while (n > 0) {
    buf[i++] = '0' + (n % 10);
    n /= 10;
  }
  while (--i >= 0) {
    printf("%c", buf[i]);
  }
}

// Spin loop checking uptime every 10000 iterations until deadline (in ticks).
// The volatile counter prevents compiler optimization. Sycall cost is amortized
// by checking uptime() only once every 10000 iterations.
static uint64
spin_until(int deadline)
{
  volatile uint64 iters = 0;
  while (1) {
    for (int i = 0; i < 10000; i++) {
      iters++;
    }
    if (uptime() >= deadline)
      break;
  }
  return iters;
}

// Spin loop executing a fixed number of target iterations (for W2 short job).
// Checks uptime() every 10000 iterations to match the instruction profile of spin_until.
static void
spin_work(uint64 target_iters)
{
  volatile uint64 iters = 0;
  while (iters < target_iters) {
    for (int i = 0; i < 10000 && iters < target_iters; i++) {
      iters++;
    }
    uptime();
  }
}

// Calibration: measure single-process spin iterations per tick over 20 ticks.
// Waits for a tick edge first to ensure a full 20-tick measurement window.
static uint64
calibrate(void)
{
  int start = uptime();
  while (uptime() == start)
    ; // align with tick edge
  start = uptime();
  int deadline = start + 20;
  uint64 iters = spin_until(deadline);
  int end = uptime();
  int elapsed = end - start;
  if (elapsed <= 0)
    elapsed = 20;
  uint64 iters_per_tick = iters / elapsed;
  printf("RESULT,calib,0,iters_per_tick,");
  print_u64(iters_per_tick);
  printf("\n");
  return iters_per_tick;
}

/*
 * Workload 1: Proportional Share
 * Fork 3 CPU-bound children with tickets 30, 20, 10.
 * Each child sets its tickets and spins until a common deadline (start + 60 ticks).
 * Under Lottery, expected shares are 30/60 (50%), 20/60 (33.3%), 10/60 (16.7%).
 * Under Round-Robin, tickets are ignored and expected shares are 1/3 each.
 * All children report their iteration counts to the parent through a shared pipe.
 */
static void
run_w1(int rep)
{
  int tickets[3] = {30, 20, 10};
  uint64 works[3] = {0, 0, 0};
  int pfd[2];
  if (pipe(pfd) < 0) {
    printf("w1: pipe failed\n");
    exit(1);
  }

  int start = uptime();
  while (uptime() == start)
    ; // align with tick edge
  start = uptime();
  int deadline = start + 60;

  for (int i = 0; i < 3; i++) {
    int pid = fork();
    if (pid < 0) {
      printf("w1: fork failed\n");
      exit(1);
    }
    if (pid == 0) {
      close(pfd[0]);
      settickets(tickets[i]);
      uint64 work = spin_until(deadline);
      struct result_msg msg;
      msg.id = i;
      msg.work = work;
      write(pfd[1], (char *)&msg, sizeof(msg));
      close(pfd[1]);
      exit(0);
    }
  }

  close(pfd[1]);
  for (int i = 0; i < 3; i++) {
    struct result_msg msg;
    read(pfd[0], (char *)&msg, sizeof(msg));
    if (msg.id >= 0 && msg.id < 3) {
      works[msg.id] = msg.work;
    }
  }
  close(pfd[0]);

  for (int i = 0; i < 3; i++) {
    wait(0);
  }

  printf("RESULT,w1,%d,work_t30,", rep);
  print_u64(works[0]);
  printf(",work_t20,");
  print_u64(works[1]);
  printf(",work_t10,");
  print_u64(works[2]);
  printf("\n");
}

/*
 * Workload 2: Turnaround Time
 * 4 CPU-bound "hog" children (10 tickets each) spin for 100 ticks.
 * 1 short job executes a fixed amount of work equal to 15 ticks of calibrated work.
 * Run two variants per repetition:
 *   (a) Short job keeps tickets = 10 (equal priority with hogs).
 *   (b) Short job calls settickets(40) (boosted priority).
 * The short job records its completion time in ticks from the common start.
 * Parent waits for all processes (short job and all 4 hogs) to finish before the next variant.
 */
static void
run_w2_variant(int rep, char variant, uint64 iters_per_tick)
{
  int short_tickets = (variant == 'a') ? 10 : 40;
  uint64 target_work = 15 * iters_per_tick;

  int pfd[2];
  if (pipe(pfd) < 0) {
    printf("w2: pipe failed\n");
    exit(1);
  }

  int start = uptime();
  while (uptime() == start)
    ; // align with tick edge
  start = uptime();
  int hog_deadline = start + 100;

  // Fork 4 hogs
  for (int i = 0; i < 4; i++) {
    int pid = fork();
    if (pid < 0) {
      printf("w2: fork hog failed\n");
      exit(1);
    }
    if (pid == 0) {
      close(pfd[0]);
      close(pfd[1]);
      settickets(10);
      spin_until(hog_deadline);
      exit(0);
    }
  }

  // Fork 1 short job
  int pid = fork();
  if (pid < 0) {
    printf("w2: fork short job failed\n");
    exit(1);
  }
  if (pid == 0) {
    close(pfd[0]);
    settickets(short_tickets);
    spin_work(target_work);
    int end = uptime();
    int turnaround = end - start;
    write(pfd[1], (char *)&turnaround, sizeof(turnaround));
    close(pfd[1]);
    exit(0);
  }

  close(pfd[1]);
  int turnaround = 0;
  read(pfd[0], (char *)&turnaround, sizeof(turnaround));
  close(pfd[0]);

  // Wait for all 5 children to finish before proceeding
  for (int i = 0; i < 5; i++) {
    wait(0);
  }

  printf("RESULT,w2,%d,variant,%c,turnaround_ticks,%d\n", rep, variant, turnaround);
}

static void
run_w2(int rep, uint64 iters_per_tick)
{
  run_w2_variant(rep, 'a', iters_per_tick);
  run_w2_variant(rep, 'b', iters_per_tick);
}

/*
 * Workload 3: Fairness
 * Fork 8 children with equal tickets (10 each) spinning until deadline (start + 60 ticks).
 * Evaluates short-term fairness across equal processes using Jain's Fairness Index
 * and the coefficient of variation.
 * All 8 children report their work counts to the parent through a shared pipe.
 */
static void
run_w3(int rep)
{
  uint64 works[8] = {0};
  int pfd[2];
  if (pipe(pfd) < 0) {
    printf("w3: pipe failed\n");
    exit(1);
  }

  int start = uptime();
  while (uptime() == start)
    ; // align with tick edge
  start = uptime();
  int deadline = start + 60;

  for (int i = 0; i < 8; i++) {
    int pid = fork();
    if (pid < 0) {
      printf("w3: fork failed\n");
      exit(1);
    }
    if (pid == 0) {
      close(pfd[0]);
      settickets(10);
      uint64 work = spin_until(deadline);
      struct result_msg msg;
      msg.id = i;
      msg.work = work;
      write(pfd[1], (char *)&msg, sizeof(msg));
      close(pfd[1]);
      exit(0);
    }
  }

  close(pfd[1]);
  for (int i = 0; i < 8; i++) {
    struct result_msg msg;
    read(pfd[0], (char *)&msg, sizeof(msg));
    if (msg.id >= 0 && msg.id < 8) {
      works[msg.id] = msg.work;
    }
  }
  close(pfd[0]);

  for (int i = 0; i < 8; i++) {
    wait(0);
  }

  printf("RESULT,w3,%d,works", rep);
  for (int i = 0; i < 8; i++) {
    printf(",");
    print_u64(works[i]);
  }
  printf("\n");
}

int
main(int argc, char *argv[])
{
  uint64 iters_per_tick = calibrate();

  // W1 proportional share (5 repetitions)
  for (int rep = 0; rep < 5; rep++) {
    run_w1(rep);
  }

  // W2 turnaround time (5 repetitions, variant a then b)
  for (int rep = 0; rep < 5; rep++) {
    run_w2(rep, iters_per_tick);
  }

  // W3 fairness (5 repetitions)
  for (int rep = 0; rep < 5; rep++) {
    run_w3(rep);
  }

  printf("BENCH_DONE\n");
  exit(0);
}
