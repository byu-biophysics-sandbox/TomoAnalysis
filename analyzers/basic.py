from analyzers import Analysis

class Basic(Analysis):
    def analyze(self, data, results, **kwargs):
        results.append({"file_path": data})
        print({"file_path": data})