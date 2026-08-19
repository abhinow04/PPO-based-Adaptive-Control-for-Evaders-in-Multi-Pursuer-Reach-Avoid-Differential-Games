import numpy as np

try:
    import ppo
    _PPO_MODULE_AVAILABLE = True
except ImportError:
    _PPO_MODULE_AVAILABLE = False


class evader:
    def __init__(self,initPos,speed,index,initOri = 0):
        self.index = index
        self.position = initPos
        self.orientation = initOri
        self.speed = speed
        self.status = 0
        self.pursuer = 0
        # Deroute multiplier m used on the most recent step (0 when no
        # deroute was applied - i.e. the direct/safe-threshold branch).
        # Tracked here so assign.py can log its evolution over the game.
        self.last_multiplier = 0.0

    def updatePos(self,position):
        
        if (self.status==0):
            position = np.array(position).flatten()[:2]
            sz_input = np.shape(position[self.index])
            sz_output = np.shape(self.position)
            #print("Eva_pos: ",position)
            if (sz_input != sz_output):
                self.position = position.T
                   
            else:
                self.position = position
            print("Evader position: ",self.position)
        else:
            print("Evader has been captured")

    def return_velocity(self, pursuer, B, tolerance, all_pursuers=None, target=None,
                         safe_threshold=120.0):
        """
        NOTE: the classical B>=0 "evader-favourable" formula is kept below
        for reference but is NOT used to drive movement - empirical testing
        showed it fails to reach the target in ~99% of cases even when B is
        nominally >= 0 (a pre-existing issue in the inherited barrier-region
        formula's sign convention, unrelated to the PPO/pursuer work done
        in this project). Instead EVERY scenario goes through the same
        proximity-gated controller: if no pursuer is within safe_threshold,
        de-route straight to the target; otherwise engage PPO.

        all_pursuers   : list of both pursuer objects used to build the
                          8-dim PPO observation [xe,ye,xp1,yp1,xp2,yp2,xt,yt]
                          described in the report. Only the assigned LP-selected
                          pursuer is used as the active threat; the other remains
                          a dummy/idle pursuer.
        target         : optional (2,) target position. Defaults to the
                          origin, matching the project plan's setup.
        safe_threshold : distance beyond which no pursuer is considered an
                          immediate threat - the evader de-routes straight
                          back onto a direct course to the target instead of
                          continuing to manoeuvre.
        """
        return self._ppo_evade(pursuer, all_pursuers, target, tolerance, safe_threshold)

    def _ppo_evade(self, pursuer, all_pursuers=None, target=None, tolerance=5.0,
                    safe_threshold=120.0):
        """Delegate to the PPO-trained evader policy for the losing (B < 0)
        scenario, but only while a pursuer is actually within safe_threshold
        of the evader. Beyond that range there's no immediate threat to
        evade, so the evader de-routes straight back onto a direct course
        to the target instead of continuing to manoeuvre - this is what
        stopped the earlier zig-zagging/dithering with no net progress.

        Falls back to the original classical fallback equation if the ppo
        module or a trained checkpoint isn't available, so the simulation
        still runs before training has happened."""
        pursuers_for_obs = all_pursuers if all_pursuers else [pursuer]
        target_pos = target if target is not None else np.zeros(2)

        # Only the specifically ASSIGNED pursuer (the `pursuer` argument) is
        # ever a real threat - assign.py's updateStatus() only checks that
        # one pursuer for capture, never "nearest of any pursuer". An
        # unassigned pursuer sitting right next to the evader is harmless.
        nearest_dist = np.linalg.norm(self.position - pursuer.position)
        if nearest_dist > safe_threshold:
            self.last_multiplier = 0.0
            to_target = target_pos - self.position
            dist_to_target = np.linalg.norm(to_target)
            if dist_to_target > 1e-6:
                return (to_target / dist_to_target) * self.speed
            return np.zeros(2)

        if _PPO_MODULE_AVAILABLE:
            try:
                velocity, m = ppo.get_ppo_velocity(
                    self.position, pursuers_for_obs, target_pos, self.speed,
                    return_multiplier=True,
                    active_pursuer_index=self.pursuer
                )
                self.last_multiplier = m
                return velocity
            except Exception as exc:
                print(f"[evader.py] PPO controller failed ({exc}); "
                      f"falling back to the classical B<0 equation.")

        # Fallback: original classical equation (heads straight for target)
        self.last_multiplier = 0.0
        xe = self.position
        alpha = self.speed / pursuer.speed
        if abs(np.linalg.norm(xe)) > 1e-2:
            gradV = (1/alpha)*(xe/np.linalg.norm(xe))
            rhoe = np.linalg.norm(gradV)
            velocity = (-self.speed/rhoe)*gradV
        else:
            velocity = np.zeros((1,2))
        return velocity


    def return_rvelocity(self, pursuer, B, tolerance):
        alpha = self.speed/pursuer.speed
        xp = pursuer.rposition
        xe = self.rposition
        xc = (xe - alpha**2 * xp)/(1 - alpha**2)
        Rc = np.linalg.norm(xc)

        
        rc = (alpha/(1-alpha**2))*np.linalg.norm(xp-xe)
        if B>=0:
            if abs(np.linalg.norm(self.rposition - pursuer.rposition))>tolerance:
                gradV = (1/(1-alpha**2))*(xc/Rc - (alpha**2/(1-alpha**2))*((xe-xp)/rc))
                rhoe = np.linalg.norm(gradV)
                velocity = (-self.speed/rhoe)*gradV

            else:
                velocity = np.zeros((1,2))

        elif (B<0):
            if abs(np.linalg.norm(xe))>1e-2:
                gradV = (1/alpha)*(xe/np.linalg.norm(xe))
                rhoe = np.linalg.norm(gradV)
                velocity = (-self.speed/rhoe)*gradV
            else:
                velocity = np.zeros((1,2))
        return velocity
