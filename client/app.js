function getRandomInclusive(min, max) {
   return Math.random() * (max - min + 1) + min;
}
var MAX = 1.5;
var MIN = -1.5;
var learningRateElement = document.getElementById('learningRate')
var iterElement = document.getElementById('iter')
var totalIterElement = document.getElementById('totalIter')
var lossElement = document.getElementById('loss')
var co_a_Element = document.getElementById('co_a')
var co_b_Element = document.getElementById('co_b')
var ctx = document.getElementById("myChart");
function makeScatterData() {
   var rs = [];
   var a = getRandomInclusive(MIN, MAX)
   for (var i = 0; i < 100; i++) {
      var x = getRandomInclusive(0, 10);
      var b = getRandomInclusive(-0.2, 0.2) + 5 - 5 * a;
      rs.push({
         x: x,
         y: a * x + b,
      });
   }
   return rs;
}
function makeLineData(coeffArray) {
   return [{
      x: -1, y: coeffArray[0] * (-1) + coeffArray[1]
   }, {
      x: 11, y: coeffArray[0] * 11 + coeffArray[1]
   }];
}
function halt() {
   unlockInput();
   isTraining = false;
}
window.iterationNumber = 2000;
window.learningRate = 0.02;
function reset() {
   unlockInput();
   window.a = tf.scalar(getRandomInclusive(MIN, MAX)).variable();
   window.b = tf.scalar(getRandomInclusive(-0.5, 0.5)).add(5).sub(a.mul(5)).variable();
   window.lineData = makeLineData([a.dataSync()[0], b.dataSync()[0]]);
   window.scatterData = [];
   window.isTraining = false;
   window.iterationNumber = iterationNumber;
   window.learningRate = learningRate;
   window.currentIterationNumber = 0;
   window.loss = '----';
   scatterData = makeScatterData()
   drawChart();
}
function onIterationNumberChange(newValue) {
   iterationNumber = newValue;
}
function onLearningRateChange(newValue) {
   learningRate = newValue;
}
function unlockInput() {
   learningRateElement.readOnly = false;
   totalIterElement.readOnly = false;
}
function lockInput() {
   learningRateElement.readOnly = true;
   totalIterElement.readOnly = true;
}
function drawIndicator() {
   learningRateElement.value = learningRate;
   totalIterElement.value = iterationNumber;
   iterElement.innerHTML = currentIterationNumber.toString();
   lossElement.innerHTML = loss.toString();
   co_a_Element.innerHTML = a.dataSync()[0].toString();
   co_b_Element.innerHTML = b.dataSync()[0].toString();
}
function drawChart(dontUpdateScatterData) {
   drawIndicator();
   if (ctx.myChart == null) {
      ctx.myChart = new Chart.Scatter(ctx, {
         data: {
            datasets: [{
               type: 'scatter',
               label: 'Dataset',
               backgroundColor: 'rgb(12, 40, 86)',
               data: scatterData,
               showLine: false
            }, {
               type: 'line', lineTension: 0,
               label: 'Interpolated Line',
               backgroundColor: 'rgb(255, 0, 0)',
               borderColor: 'rgb(255, 0, 0)',
               data: lineData,
               fill: false,
               showLine: true
            }]
         },
         options: {
            responsive: false,
            scales: {
               xAxes: [{
                  type: 'linear',
                  position: 'bottom',
                  ticks: {
                     min: 0, max: 10
                  }
               }],
               yAxes: [{
                  type: 'linear',
                  ticks: {
                     min: 0, max: 10
                  }
               }]
            }
         }
      });
   }
   else {
      dontUpdateScatterData !== true && (ctx.myChart.data.datasets[0].data = scatterData);
      ctx.myChart.data.datasets[1].data = lineData;
      ctx.myChart.update();
   }
}
function train() {
   if (isTraining) return;
   isTraining = true;
   lockInput();
   const optimizer = tf.train.sgd(learningRate)
   function predict(x) {
      return tf.tidy(() => {
         return a.mul(x).add(b);
      });
   }
   function calculateLoss(pred, labels) {
      return pred.sub(labels).square().mean();
   }
   function _train(x, y, iterationNumber, i) {
      if (isTraining == false) return;
      if (i >= iterationNumber) {
         isTraining = false;
         return;
      }
      currentIterationNumber = i + 1;
      optimizer.minimize(function () {
         var stepLoss = calculateLoss(predict(x), y);
         loss = stepLoss.dataSync()[0]
         return stepLoss;
      });
      lineData = makeLineData([a.dataSync()[0], b.dataSync()[0]]);
      drawChart(true);
      window.requestAnimationFrame(function () {
         _train(x, y, iterationNumber, i + 1);
      });
   }
   var xVector = [];
   var yVector = [];
   for (var i = 0; i < scatterData.length; i++) {
      xVector.push(scatterData[i].x);
      yVector.push(scatterData[i].y);
   }
   xVector = tf.tensor1d(xVector);
   yVector = tf.tensor1d(yVector);
   _train(xVector, yVector, iterationNumber, 0);
}