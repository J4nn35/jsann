# how to use it

First, clone the repository:
```bash
git clone https://github.com/aderthecoder/JSANN.git
```
then type this code:
```js
const Network = require("./network");

numberOfInputNeurons = 1;
numberOfOutputNeurons = 1;
numberOfHiddenlayers = 2;
numberOfHiddenNeurons = 2;
const net = new Network(numberOfInputNeurons, numberOfOutputNeurons, numberOfHiddenlayers, numberOfHiddenNeurons);
//train it
const trainingData = [[1, 2], [2, 4], [3, 6]];
for(const data of trainingData) {
    net.train([data[0]], [data[1]]);
}
//then use it
const testData = [1, 2, 3]
for(const data of testData) {
    console.log(`Expected ${2 * data}, got ${net.forward([data])}`);
}

```
