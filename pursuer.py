import numpy as np

class pursuer:
    def __init__(self, initPos, speed, index, initOri=0):
        self.index = index
        self.position = initPos
        self.speed = speed
        self.status = 0
        self.self = 0

    def reset_lock(self):
        """Not used in continuous-tracking mode, kept for compatibility."""
        pass

    def updatePos(self, position):
        position = np.array(position).flatten()[:2]
        try:
            sz_input = np.shape(position[self.index])
        except:
            sz_input = np.shape(position[0][self.index])
        sz_output = np.shape(self.position)
        if sz_input != sz_output:
            self.position = position.T
        else:
            self.position = position
        print("Pursuer position: ", self.position)

    def return_velocity(self, evader, B, tolerance):
        """Continuously track the evader's live position every step.
        Recompute heading each call based on current evader location."""
        alpha = evader.speed / self.speed
        xe = np.array(evader.position).flatten()
        xp = np.array(self.position).flatten()

        # Head directly toward wherever the evader is right now
        direction = xe - xp
        norm = np.linalg.norm(direction)
        heading = direction / norm if norm > 1e-9 else np.zeros(2)

        if np.linalg.norm(xe - xp) > tolerance:
            velocity = self.speed * heading
        else:
            velocity = np.zeros((2,))
        return velocity