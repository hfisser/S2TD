class ProgressBar:
    def __init__(self, max_value, modulo_threshold):
        """
        This is a progress bar that can be printed during iterative processes of fixed length.
        :param max_value: int the last value of the iteration, usually an index.
        :param modulo_threshold: int modulo that determines how often the progress bar is actually updated.
        """
        self.max_value = max_value
        self.modulo_threshold = modulo_threshold

    def update(self, i):
        """
        Updates the progress bar console print according to the current iteration value.
        :param i: int current value of the iteration, usually an index, always <= max_value.
        :return: None
        """
        last = (i + self.modulo_threshold - 1) > self.max_value
        if (i % self.modulo_threshold) == 0 or last:
            max_print = 35
            i += 1
            n = int((i / self.max_value) * 100)
            n_symbols = int(max_print * (n / 100))
            spaces = "-" * int(max_print - n_symbols)
            i = self.max_value if (i + 1) == self.max_value else i
            n_str = str(i) + "/" + str(self.max_value) + "|"
            print("\r", "|" + "=" * n_symbols + spaces + "| |" + str(n) + " %| |" + n_str, end=" ")
