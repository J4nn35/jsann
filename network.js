function Network(inputNeurons, outputNeurons, hiddenLayers = 0, hiddenNeurons = 0, learningRate = 0.001) {

    //cell[L][n] = a_n^(L)
    //the value of the n-th neuron at the L-th layer
    this.neurons = [];
    //weights[L][j][k] = w_kj^(L) (pass value of a_j^(L-1) to a_k^(L))
    //the value of the weight connects k-th neuron at L-th layer and j-th neuron at L-1-th layer
    this.weights = [undefined];
    //biases[L][n] = b_n^(L)
    //the bias of the n-th neuron at the L-th layer
    this.biases = [undefined];
    //z[L][n] = z_n^(L)
    //the weighted value of the n-th neuron at L-th layer
    this.z = [undefined];
    //the learning rate of the ANN
    this.learningRate = learningRate;

    //initializing values of this ANN
    let layer = [];
    let bias = [];
    let zl = [];
    for(let n = 0; n < inputNeurons; n++) {
        layer.push(0);
    }
    this.neurons.push(layer);
    layer = [];

    for(let L = 0; L < hiddenLayers; L++) {
        for(let n = 0; n < hiddenNeurons; n++) {
            layer.push(0);
            bias.push(Math.random() - 0.5);
            zl.push(0);
        }
        this.neurons.push(layer);
        this.biases.push(bias);
        this.z.push(zl);
        bias = [];
        layer = [];
        zl = [];
    }

    for(let n = 0; n < outputNeurons; n++) {
        layer.push(0);
        bias.push(Math.random() - 0.5);
        zl.push(0);
    }
    this.neurons.push(layer);
    this.biases.push(bias);
    this.z.push(zl);
    bias = [];
    layer = [];
    zl = [];

    this.L = this.neurons.length;

    wj = [];
    wk = [];
    for(let l = 1; l < this.L; l++) {
        for(let j = 0; j < this.neurons[l - 1].length; j++) {
            for(let k = 0; k < this.neurons[l].length; k++) {
                wk.push(Math.random() * 4 - 2);
            }
            wj.push(wk);
            wk = [];
        }
        this.weights.push(wj);
        wj = [];
    }

}

//activation function
Network.activation = function(x) {
    return x > 0 ? 0.5 * x : 0.01 * x;
};

//derivative of the activation function
Network.dactivation = function(x) {
    return x > 0 ? 0.5 : 0.01;
};

//cost function
Network.cost = function(outputs, y) {
    let res = 0;
    for(let i = 0; i < y.length; i++) {
        res += Math.pow(outputs[i] - y[i], 2);
    }
    return 0.5 * y.length * res;
}

//derivative of the cost function
Network.dcost = function(outputs, y) {
    return outputs - y;
}

Network.prototype = {

    //forward propogate the network
    forward: function(inputs) {

        if(inputs.length !== this.neurons[0].length) { throw new Error("input and the input layer must have same length"); }
        this.neurons[0] = inputs;

        //a[l-1][pren] * w[l][pren][n] + b[l][n] (pren: number of neurons of previous layer)
        //calculating weighted value of n-th neuron at l-th layer
        for(let l = 1; l < this.L; l++) {
            for(let n = 0; n < this.neurons[l].length; n++) {
                this.z[l][n] = 0
                for(let pren = 0; pren < this.neurons[l - 1].length; pren++) {
                    this.z[l][n] += this.neurons[l - 1][pren] * this.weights[l][pren][n];
                }
                this.z[l][n] += this.biases[l][n];
                this.neurons[l][n] = Network.activation(this.z[l][n]);
            }
        }
        return this.neurons[this.L - 1];
    },

    //backward propogate the ANN (a.k.a learning)
    backward: function(outputs) {
        
        if(outputs.length !== this.neurons[this.L - 1].length) { throw new Error("output and output layer must have same length"); }

        //initilize the derivative of neurons
        let dcda = [undefined];
        let dcdal = [];
        for(let i = 1; i < this.L; i++) {
            for(let j = 0; j < this.neurons[i].length; j++) {
                dcdal.push(0);
            }
            dcda.push(dcdal);
            dcdal = [];
        }
        
        //calculate the derivative of neurons
        for(let n = 0; n < this.neurons[this.L - 1].length; n++) {
            dcda[this.L - 1][n] = Network.dcost(this.neurons[this.L - 1][n], outputs[n]);
        }
        for(let l = this.L - 2; l > 0; l--) {
            for(let n = 0; n < this.neurons[l].length; n++) {
                dcda[l][n] = 0;
                for(let k = 0; k < this.neurons[l + 1].length; k++) {
                    dcda[l][n] += this.weights[l + 1][n][k] * Network.dactivation(this.z[l + 1][k]) * dcda[l + 1][k];
                }
            }
        }
        
        //initilize the derivative of weights
        let dcdw = [undefined];
        let dcdwj = [];
        let dcdwk = [];
        for(let i = 1; i < this.weights.length; i++) {
            for(let j = 0; j < this.weights[i].length; j++) {
                for(let k = 0; k < this.weights[i][j].length; k++) {
                    dcdwk.push(0);
                }
                dcdwj.push(dcdwk);
                dcdwk = [];
            }
            dcdw.push(dcdwj);
            dcdwj = [];
        }
        
        //initilize the derivative of biases
        let dcdb = [undefined];
        let dcdbl = [];
        for(let i = 1; i < this.biases.length; i++) {
            for(let j = 0; j < this.biases[i].length; j++) {
                dcdbl[j] = 0;
            }
            dcdb[i] = dcdbl;
            dcdbl = [];
        }
        
        //calculate the derivative of weights and biases
        for(let l = this.L - 1; l > 0; l--) {
            for(let n = 0; n < this.neurons[l].length; n++) {
                for(let k = 0; k < this.neurons[l - 1].length; k++) {
                    dcdw[l][k][n] = this.neurons[l - 1][k] * Network.dactivation(this.z[l][n]) * dcda[l][n];
                    this.weights[l][k][n] -= this.learningRate * dcdw[l][k][n];
                }
                dcdb[l][n] = Network.dactivation(this.z[l][n]) * dcda[l][n];
                this.biases[l][n] -= this.learningRate * dcdb[l][n];
            }
        }

        return Network.cost(this.neurons[this.L - 1], outputs);
        
    },

    train: function(inputs, outputs) {
        this.forward(inputs);
        return this.backward(outputs);
    },

    //TODO: save, load

};

module.exports = Network;