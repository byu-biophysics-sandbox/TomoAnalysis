from writers import Writer

class Excel(Writer):
    def __init__(self):
        super().__init__()

    def write(self, batch_data):
        self._create_dataframe()

        super().write()

if __name__ == "__main__":
    writer = Excel()
    writer.write()