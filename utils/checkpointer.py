
class Checkpointer:
    def __init__(
            self,
            checkpoint_path,
            interval
        ):
        self.checkpoint_path = checkpoint_path
        self.interval = interval

    def checkpoint(
        self,
        current_batch
    ):
        pass