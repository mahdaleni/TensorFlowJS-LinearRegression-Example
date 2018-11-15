# TensorFlowJS Linear Regression Example
An example of how to use TensorFlowJS's Low-level API to do Linear Regression

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

(The corresponding tensorflowjs api is tf.train.sgd, stands for Stochastic gradient descent, but it's just a normal gradient descent optimizer despite it's name)