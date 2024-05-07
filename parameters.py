import yaml

class parameterCombinations():
    def __init__(self, fileName):
        with open(fileName, 'r') as f:
            self.parameters = yaml.load(f.read(), Loader = yaml.FullLoader)
        f.close()

        self.binaryconfig = {}
        self.parametersList = list(self.parameters.keys())
        assert len(self.parametersList) > 0

    def __call__(self):
        for ele in self.searchParameters(self.parametersList[0]):
            yield ele

    def nextKey(self, thisKey):
        index = self.parametersList.index(thisKey)
        if index + 1 < len(self.parametersList):
            return self.parametersList[index + 1]
        return "#"

    def searchParameters(self, thisKey):
        if thisKey == "#":
            yield self.binaryconfig
        elif isinstance(self.parameters[thisKey], list):
            for item in self.parameters[thisKey]:
                self.binaryconfig[thisKey] = item
                for ele in self.searchParameters(self.nextKey(thisKey)):
                    yield ele
        else:
            self.binaryconfig[thisKey] = self.parameters[thisKey]
            for ele in self.searchParameters(self.nextKey(thisKey)):
                yield ele

    def getPrefix(self):
        prefix = []
        for parameter in self.parametersList:
            if isinstance(self.parameters[parameter], list):
                prefix.append(parameter)
        return prefix
