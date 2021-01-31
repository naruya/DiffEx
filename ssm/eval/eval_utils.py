import re


class Extractor(object):
    def __init__(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        self.lines = {}
        for mode in ["train", "test"]:
            key = "({}) Epoch".format(mode)
            self.lines[mode] = [line for line in lines if key in line]

    def __call__(self, mode, key):
        key = "\'{}\'".format(key)
        lines = self.lines[mode]
        lines = [re.split('[{}]', line)[1].split(', ') for line in lines]
        lines = [[item for item in line if key in item] for line in lines]
        lines = [float(line[0].split()[1]) for line in lines]
        return lines