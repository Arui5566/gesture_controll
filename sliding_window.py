class SlidingWindow:
    def __init__(self, window_size=8, stable_threshold=5):
        self.window_size = window_size
        self.stable_threshold = stable_threshold
        self.window = []
        self.stable_output = None

    def add(self, label):
        self.window.append(label)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def get_stable(self):
        if not self.window:
            return None

        counts = {}
        for item in self.window:
            counts[item] = counts.get(item, 0) + 1

        stable = max(counts, key= counts.get)

        if counts[stable] >= self.stable_threshold:
            self.stable_output = stable

        return self.stable_output
