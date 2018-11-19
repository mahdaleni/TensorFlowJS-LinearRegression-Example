# TensorFlowJS Linear Regression and Polynominal Regression Examples
Examples of how to use TensorFlowJS's Low-level API to do Linear Regression and Polynomial Regression

### Installing and Running

Try it online: https://meigyoku-thmn.github.io/TensorFlowJS-LinearRegression-Example/client/ <br />
Or clone this repo then run this command in terminal:
```bash
npm i
node index
```
Then go to http://localhost:3000

### Explaination
This page do the following: <br />

1. Randomly scatter 100 dots in the graph
2. Draw a line with random coefficients
3. Use gradient descent algorithm to modify line's coefficients until Total distance of all dots to the line (on yAxis) is minimum. <br />

### Note 

- The used tensorflowjs api is tf.train.sgd, stands for Stochastic Gradient Descent, but it's just a normal gradient descent optimizer despite it's name);
- The convergence speed will be super fast if you can make your training data range length (input) approximately 2 (eg: x ∈ [-1, 1]) and learning rate 0.5.