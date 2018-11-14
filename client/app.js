function getRandomInclusive(min, max) {
   return Math.random() * (max - min + 1) + min;
}
function makeScatterData() {
   var rs = [];
   var a = Math.random();
   for (var i = 0; i < 100; i++) {
      var x = getRandomInclusive(0, 10);
      var b = getRandomInclusive(0, 1);
      rs.push({
         x: x,
         y: a * x + b
      });
   }
   return rs;
}
function makeLineData(coeffArray) {
   return [{
      x: 0, y: coeffArray[0] * 0 + coeffArray[1]
   }, {
      x: 10, y: coeffArray[0] * 10 + coeffArray[1]
   }];
}
function reset() {
   window.a = tf.variable(tf.scalar(Math.random()))
   window.b = tf.variable(tf.scalar(Math.random()))
   window.lineData = makeLineData([a.dataSync()[0], b.dataSync()[0]]);
   window.scatterData = [];
   window.isTraining = false;
   scatterData = makeScatterData()
   drawChart();
}
function drawChart(dontUpdateScatterData) {
   var ctx = document.getElementById("myChart");
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
   const iterationNumber = 2000;
   const learningRate = 0.001;
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
      optimizer.minimize(function () {
         var stepLoss = calculateLoss(predict(x), y);
         console.log({
            loss: stepLoss.dataSync()[0],
            a: a.dataSync()[0],
            b: b.dataSync()[0],
         })
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