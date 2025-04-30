class GridWorldEnv:
    """
    GridWorld using (row, col) coordinate system:
    - Action 0 = up    → row -= 1
    - Action 1 = right → col += 1
    - Action 2 = down  → row += 1
    - Action 3 = left  → col -= 1
    """

    def __init__(self, rows=5, cols=5, obstacles=None):
        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles or set()
        self.start_state = (0, 0)
        self.goal_state = (4, 4)
        self.state = None
        self.action_space = [0, 1, 2, 3]  # up, right, down, left

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 0:  # up
            r = max(r - 1, 0)
        elif action == 1:  # right
            c = min(c + 1, self.cols - 1)
        elif action == 2:  # down
            r = min(r + 1, self.rows - 1)
        elif action == 3:  # left
            c = max(c - 1, 0)
        next_state = (r, c)
        if next_state in self.obstacles:
            next_state = self.state  # blocked
        self.state = next_state
        if self.state == self.goal_state:
            return self.state, 1.0, True, {}
        else:
            return self.state, -0.01, False, {}
