import numpy as np
from scipy.optimize import linprog
import pandas as pd
from evader import evader
from pursuer import pursuer
import matplotlib.pyplot as plt
import gogoal


class assignment:
    def __init__(self, pursuer_position, evader_position, target_position, speed, pur_sp, eva_sp):

        self.pur_pos = np.array(pursuer_position)   # (n, 2)
        self.eva_pos = np.array(evader_position)    # (m, 2)
        self.n = len(self.pur_pos)                  # number of pursuers
        self.m = len(self.eva_pos)                  # number of evaders
        self.pur_sp = pur_sp                        # (n, 1)
        self.eva_sp = eva_sp                        # (m, 1)
        self.mpn = self.m + self.n
        self.mn  = self.m * self.n
        self.cost = np.zeros([self.mn, self.mpn])
        self.tolerance = 5
        self.timestep  = 0.1
        self.target    = target_position
        self.val       = np.zeros([self.m, self.n])  # (m, n)
        self.B         = np.zeros([self.m, self.n])  # (m, n)
        self.assigned  = None


        self.alpha = self.eva_sp.reshape(self.m, 1) / self.pur_sp.reshape(1, self.n)  # (m, n)

        self.pursuers = [None] * self.n
        self.evaders  = [None] * self.m

        for i in range(self.n):
            self.pursuers[i] = pursuer(self.pur_pos[i, :].T, self.pur_sp[i], i)

        for j in range(self.m):
            self.evaders[j] = evader(self.eva_pos[j, :].T, self.eva_sp[j], j)

    # ── Barrier region ────────────────────────────────────────────────────────
    def barrier_region(self):
        # BUG 5 FIX: ||x||^2 = sum(x^2), not norm(x*x).
        # np.linalg.norm(x*x, 2, 1) is the L2 norm of the squared components — wrong.
        nes = np.sum(self.eva_pos ** 2, axis=1)  # (m,)  = ||xe_j||^2
        nps = np.sum(self.pur_pos ** 2, axis=1)  # (n,)  = ||xp_i||^2

        # B[j, i] = ||xe_j||^2 - alpha[j,i]^2 * ||xp_i||^2
        # nes[:, None] → (m, 1);  nps[None, :] → (1, n);  alpha**2 → (m, n)
        self.B = nes[:, None] - (self.alpha ** 2) * nps[None, :]  # (m, n)

    # ── Value matrix ──────────────────────────────────────────────────────────
    def val_mat(self):
        nps = np.sum(self.pur_pos ** 2, axis=1)  # (n,)
        nes = np.sum(self.eva_pos ** 2, axis=1)  # (m,)

        psh = self.pur_pos.shape   # (n, 2)
        esh = self.eva_pos.shape   # (m, 2)

        # dist1[i, j] = ||xp_i - xe_j||   shape: (n, m)
        dist1 = np.sqrt(np.sum(
            (self.pur_pos[:, None, :] - self.eva_pos[None, :, :]) ** 2,
            axis=2))                                                   # (n, m)

        # v1[j, i] — used when alpha == 1
        # 0.5 * (||xe_j||^2 - ||xp_i||^2 / dist(xp_i, xe_j))
        v1 = 0.5 * (nes[None, :] - nps[:, None] / np.maximum(dist1, 1e-9)).T  # (m, n)

        # difference[j, i, :] = xe_j - xp_i
        difference = self.eva_pos[:, None, :] - self.pur_pos[None, :, :]  # (m, n, 2)
        dist2 = np.sqrt(np.sum(difference ** 2, axis=2))                  # (m, n)

        alphasq      = self.alpha ** 2                  # (m, n)
        alpha_factor = self.alpha / (1 - alphasq)       # (m, n)

        # BUG 2 FIX: first_term must stay (m, n).
        # Original code used np.sqrt(np.sum(...)) with no axis= → collapsed to scalar.
        # Correct: sum only over the spatial axis (axis=2).
        #
        # cap[j, i, :] = xe_j - alpha[j,i]^2 * xp_i
        cap = (self.eva_pos[:, None, :]                          # (m, 1, 2)
               - alphasq[:, :, None] * self.pur_pos[None, :, :])  # (m, n, 2)
        first_term  = np.sqrt(np.sum(cap ** 2, axis=2))           # (m, n)
        second_term = alpha_factor * dist2                         # (m, n)
        v2 = first_term - second_term                              # (m, n)

        # v3[j, i] — used when B < 0
        v3 = -np.sqrt(nps[None, :]) + np.sqrt(nes[:, None]) / self.alpha  # (m, n)

        # Index masks — all shape (m, n)
        idx1 = (self.B >= 0) & (self.alpha == 1)
        idx2 = (self.B >= 0) & (self.alpha <  1)
        idx3 = (self.B <  0) & (self.alpha <= 1)

        self.val = idx1 * v1 + idx2 * v2 + idx3 * v3  # (m, n)

        idx    = (self.B >= 0) & (self.alpha <= 1)
        self.a = idx.astype(float) * self.val          # (m, n)

        # BUG 4 FIX: removed hardcoded print(self.a[1]) which crashes when m == 1.
        print("Value matrix a:\n", self.a)

        self.linporgram()

    # ── Linear program ────────────────────────────────────────────────────────
    def linporgram(self):


        f = self.a.T.flatten()  # (n*m,) — full objective, not just 2 elements

        b = np.ones([self.mpn, 1])
        A = np.zeros([self.mpn, self.mn])

        # Each pursuer assigned to at most 1 evader: rows 0..n-1
        A[:self.n, :] = np.tile(np.eye(self.n), (1, self.m))

        # Each evader assigned to at most 1 pursuer: rows n..n+m-1
        row_indices      = np.arange(self.n, self.n + self.m)
        col_start_indices = (row_indices - self.n) * self.n
        col_indices      = np.arange(self.n) + col_start_indices[:, np.newaxis]
        A[row_indices[:, np.newaxis], col_indices] = 1

        bounds = [(0, None)] * self.mn

        if self.n >= self.m:
            result = linprog(-f,
                             A_ub=A[:self.n,      :], b_ub=b[:self.n],
                             A_eq=A[self.n:self.mpn, :], b_eq=b[self.n:self.mpn],
                             bounds=bounds, method='highs')
        else:
            result = linprog(-f,
                             A_ub=A[self.n:self.mpn, :], b_ub=b[self.n:self.mpn],
                             A_eq=A[:self.n,      :], b_eq=b[:self.n],
                             bounds=bounds, method='highs')

        if not result.success:
            print("WARNING: linprog failed:", result.message)
            return

        self.x = np.reshape(result.x, [self.n, self.m]).T  # (m, n)

    # ── Check win condition ───────────────────────────────────────────────────
    def check_win(self, live_plot=True):
        self.barrier_region()
        self.val_mat()
        self.linporgram()

        [row, columns] = np.where(self.x == 1)

        for i in range(len(row)):
            self.evaders[row[i]].pursuer = columns[i]

        for j in range(len(columns)):
            self.pursuers[columns[j]].evader = row[j]

        check     = self.a * self.x
        check_min = min(np.reshape(check, (len(check) * len(check[0]), 1)))

        self.win = 1 if check_min[0] < 0 else 0

        mask = (self.alpha > 1) * self.x
        if mask.any():
            self.check_cond = 1
            return None
        else:
            self.check_cond = 0
            return self.plot_contin(live_plot=live_plot)



    # ── Simulation loop ───────────────────────────────────────────────────────
    def plot_contin(self, live_plot=True):
        done       = False
        t          = 0
        time_chunk = 1000
        pursuer_traj = np.zeros((self.n, time_chunk, 2))
        evader_traj  = np.zeros((self.m, time_chunk, 2))
        # distance history: dist_to_pursuer[j, t] = distance from evader 0 to
        # pursuer j at step t; dist_to_target[t] = distance from evader 0 to
        # the target at step t. Assumes the single-evader case (self.m == 1),
        # which matches this project's setup.
        dist_to_pursuer = np.zeros((self.n, time_chunk))
        dist_to_target  = np.zeros(time_chunk)
        multiplier_history = np.zeros(time_chunk)  # evolution of the deroute multiplier m

        while not done:
            pursuer_traj[:, t, :] = np.array([p.position.T for p in self.pursuers])
            evader_traj[:, t, :]  = np.array([e.position.T for e in self.evaders])

            ev_pos = self.evaders[0].position
            for j in range(self.n):
                dist_to_pursuer[j, t] = np.linalg.norm(ev_pos - self.pursuers[j].position)
            dist_to_target[t] = np.linalg.norm(ev_pos - self.target)

            if live_plot:
                plt.figure("Live Pursuer-Evader Trajectory")
                plt.clf()

                active_idx = self.evaders[0].pursuer if self.evaders else 0
                for j in range(self.n):
                    is_active = (j == active_idx)
                    line_color = 'r' if is_active else 'k'
                    label = "Pursuer "+ str(j+1)
                    plt.plot(pursuer_traj[j, :t+1, 0], pursuer_traj[j, :t+1, 1],
                             line_color, label=label)
                    plt.scatter(self.pursuers[j].position[0], self.pursuers[j].position[1],
                                color=line_color, marker='o', s=100)

                for i in range(self.m):
                    plt.plot(evader_traj[i, :t+1, 0], evader_traj[i, :t+1, 1],
                             'b', label="Evader" if i == 0 else "")
                    plt.scatter(self.evaders[i].position[0], self.evaders[i].position[1],
                                color='b', marker='x', s=100)

                plt.scatter(0, 0, color='g', s=100, label="Target")
                plt.xlabel("X-axis")
                plt.ylabel("Y-axis")
                plt.title("Live Pursuer-Evader Trajectory")
                plt.legend()
                plt.grid(True)
                plt.pause(0.001)

                E_statuses = []
                for i, e in enumerate(self.evaders):
                    if e.status == 0:
                        E_statuses.append(f"{i+1}: Not captured")
                    elif e.status == 1:
                        E_statuses.append(f"{i+1}: Captured")
                    elif e.status == 2:
                        E_statuses.append(f"{i+1}: Target Breached")

                print("\n\n--- Updating Status ---")
                print(f"Step {t}:\n{E_statuses}")

            if all(e.status == 1 for e in self.evaders):
                if live_plot:
                    print("All evaders captured. Terminating simulation.")
                break

            done = self.step()
            multiplier_history[t] = self.evaders[0].last_multiplier
            t += 1

            if t >= pursuer_traj.shape[1]:
                pursuer_traj = np.concatenate(
                    (pursuer_traj, np.zeros((self.n, time_chunk, 2))), axis=1)
                evader_traj = np.concatenate(
                    (evader_traj, np.zeros((self.m, time_chunk, 2))), axis=1)
                dist_to_pursuer = np.concatenate(
                    (dist_to_pursuer, np.zeros((self.n, time_chunk))), axis=1)
                dist_to_target = np.concatenate(
                    (dist_to_target, np.zeros(time_chunk)))
                multiplier_history = np.concatenate(
                    (multiplier_history, np.zeros(time_chunk)))

        if not live_plot:
            outcome = "captured" if self.evaders[0].status == 1 else (
                "reached_target" if self.evaders[0].status == 2 else "timeout")
            return outcome

        self._plot_distance_graphs(dist_to_pursuer[:, :t+1], dist_to_target[:t+1])
        self._plot_end_of_game_graphs(
            pursuer_traj[:, :t+1, :], evader_traj[:, :t+1, :],
            multiplier_history[:t+1], self.timestep)
        plt.show()

    def _plot_distance_graphs(self, dist_to_pursuer, dist_to_target):
        """Plots distance-to-pursuer-1, distance-to-pursuer-2 (etc, one per
        pursuer), and distance-to-target, each against time step, in a
        separate figure alongside the live trajectory plot."""
        n_pursuers = dist_to_pursuer.shape[0]
        timesteps = np.arange(dist_to_pursuer.shape[1])

        fig, axes = plt.subplots(n_pursuers + 1, 1, figsize=(7, 3 * (n_pursuers + 1)),
                                  num="Distance Over Time", sharex=True)
        if n_pursuers + 1 == 1:
            axes = [axes]

        for j in range(n_pursuers):
            axes[j].plot(timesteps, dist_to_pursuer[j], color='r')
            axes[j].set_ylabel("Distance")
            axes[j].set_title(f"Distance to Pursuer {j+1} Over Time")
            axes[j].grid(True)

        axes[-1].plot(timesteps, dist_to_target, color='g')
        axes[-1].set_ylabel("Distance")
        axes[-1].set_xlabel("Time step")
        axes[-1].set_title("Distance to Target Over Time")
        axes[-1].grid(True)

        fig.tight_layout()

    def _plot_end_of_game_graphs(self, pursuer_traj, evader_traj, multiplier_history, dt):
        """Plots, each in its own figure with title + legend:
        1. Evolution of the deroute multiplier m over time
        2. Angular velocity of all agents over time
        """
        T = pursuer_traj.shape[1]
        timesteps = np.arange(T)

        # ---- 1. Evolution of m ----
        plt.figure("Evolution of m")
        plt.plot(timesteps, multiplier_history, color='purple', label="Deroute multiplier m")
        plt.xlabel("Time step")
        plt.ylabel("m")
        plt.title("Evolution of Deroute Multiplier (m) Over Time")
        plt.legend()
        plt.grid(True)

        # ---- helper: velocity / speed / heading from a position trajectory ----
        def velocities(traj):
            # traj: (T, 2) -> (T-1, 2) finite-difference velocity
            return np.diff(traj, axis=0) / dt

        # Compute headings for pursuers and evader (needed for angular velocity)
        pursuer_headings = []
        for j in range(self.n):
            vel = velocities(pursuer_traj[j])
            pursuer_headings.append(np.arctan2(vel[:, 1], vel[:, 0]))

        vel_e = velocities(evader_traj[0])
        evader_heading = np.arctan2(vel_e[:, 1], vel_e[:, 0])

        # ---- 2. Angular velocity of the agents ----
        plt.figure("Angular Velocity")

        def angular_velocity(heading):
            # unwrap to avoid +/-pi wraparound spikes, then finite-difference
            unwrapped = np.unwrap(heading)
            return np.diff(unwrapped) / dt

        omega_e = angular_velocity(evader_heading)
        plt.plot(timesteps[2:], omega_e, color='b', label="Evader")
        for j in range(self.n):
            omega_p = angular_velocity(pursuer_headings[j])
            plt.plot(timesteps[2:], omega_p, label=f"Pursuer {j+1}")
        plt.xlabel("Time step")
        plt.ylabel("Angular velocity (rad/s)")
        plt.title("Angular Velocity of Agents Over Time")
        plt.legend()
        plt.grid(True)
    def updateStatus(self):
        for i, ev in enumerate(self.evaders):
            j = ev.pursuer
            if j is not None and isinstance(j, (int, np.integer)):
                pu       = self.pursuers[j]
                distance  = np.linalg.norm(pu.position - ev.position)
                distance1 = np.linalg.norm(ev.position - self.target)

                if distance < self.tolerance:
                    ev.status = 1
                    pu.status = 1
                elif distance1 < self.tolerance:
                    ev.status = 2
                    pu.status = 1



    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self):
        self.updateStatus()
        print("Step: Updating positions")

        if all(e.status == 1 for e in self.evaders):
            print("All evaders captured. Stopping simulation.")
            return True

        if any(e.status == 2 for e in self.evaders):
            print("Evader has breached the target point.")
            return True

        for i, ev in enumerate(self.evaders):
            if ev.status == 0:
                j = np.where(self.x[i, :] == 1)[0]
                if j.size > 0:
                    eva_retVel = ev.return_velocity(
                        self.pursuers[j[0]], self.B[i, j[0]], self.tolerance,
                        all_pursuers=self.pursuers, target=self.target)
                    ev.updatePos(ev.position + self.timestep * eva_retVel)

        for j, pu in enumerate(self.pursuers):
            if pu.status == 0:
                i = np.where(self.x[:, j] == 1)[0]
                if i.size > 0:
                    self.assigned = i[0]
                    pur_retVel = pu.return_velocity(
                        self.evaders[i[0]], self.B[i[0], j], self.tolerance)
                    pu.updatePos(pu.position + self.timestep * pur_retVel)

        return False


# ── Batch trials: win rate over time (across many games) ───────────────────
def run_batch_trials(num_trials=100, n=2, m=1, bounds=300, target_bound=200,
                      pur_speed=30, eva_speed=18, seed=0):
    """
    A single game only has one outcome (win/loss) - there's no 'rate' within
    one run. This runs many independent games headlessly (no live plotting)
    and produces a scatter of each trial's outcome (1=win, 0=loss) against
    trial number, with a rolling win-rate line overlaid, titled and
    legended, to show how win rate evolves across repeated trials.
    """
    rng = np.random.default_rng(seed)
    outcomes = []

    for trial in range(num_trials):
        pur_pos = bounds * rng.standard_normal((n, 2))
        ev_pos = target_bound * rng.standard_normal((m, 2))
        target = np.array([0, 0])
        pur_sp = np.array([[pur_speed]] * n)
        eva_sp = np.array([[eva_speed]] * m)

        asgn = assignment(pur_pos, ev_pos, target, np.ones(m), pur_sp, eva_sp)
        outcome = asgn.check_win(live_plot=False)
        win = 1 if outcome == "reached_target" else 0
        outcomes.append(win)
        print(f"Trial {trial+1}/{num_trials}: {'WIN' if win else 'loss'}")

    outcomes = np.array(outcomes, dtype=float)
    trial_idx = np.arange(1, num_trials + 1)

    def rolling_mean(x, w):
        w = min(w, len(x))
        return np.convolve(x, np.ones(w) / w, mode="valid")

    window = max(1, min(20, num_trials // 5))
    roll = rolling_mean(outcomes, window)
    roll_x = trial_idx[window - 1:]

    plt.figure("Win Rate Over Time (Trials)")
    plt.scatter(trial_idx, outcomes, color='tab:blue', label="Trial outcome (1=win, 0=loss)")
    plt.plot(roll_x, roll, color='tab:red', label=f"Rolling win rate ({window}-trial window)")
    plt.xlabel("Trial number")
    plt.ylabel("Outcome / Win rate")
    plt.title("Win Rate Over Time (Across Trials)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Overall win rate: {outcomes.mean():.2%} ({int(outcomes.sum())}/{num_trials})")
    return outcomes

