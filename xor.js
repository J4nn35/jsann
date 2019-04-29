//creates a net that simulates a xor gate

const Network = require("./network");

const xor = new Network(2, 1, 5, 5, 1);

for(let i = 0; i <= 10000; i++) {
    
    let b1 = Math.round(Math.random());
    let b2 = Math.round(Math.random());
    let data = {
        inputs: [b1, b2],
        outputs: [b1 ^ b2]
    };

    xor.train(data);

}

const test = [[0, 0], [0, 1], [1, 0], [1, 1]];

for(let i = 0; i < test.length; i++) {
    let output = xor.forward(test[i]);
    let y = test[i][0] ^ test[i][1];
    console.log(`expected: ${y}, ai: ${output}, cost:${Math.pow(output - y, 2)}`)
}