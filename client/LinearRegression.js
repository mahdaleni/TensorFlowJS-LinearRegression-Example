var MAX = 1.5;
var MIN = -1.5;
function LinearRegression(frame) {
   this.learningRateElement = frame.getElementsByClassName('learningRate')[0]
   this.iterElement = frame.getElementsByClassName('iter')[0]
   this.totalIterElement = frame.getElementsByClassName('totalIter')[0]
   this.lossElement = frame.getElementsByClassName('loss')[0]
   this.co_a_Element = frame.getElementsByClassName('co_a')[0]
   this.co_b_Element = frame.getElementsByClassName('co_b')[0]
   this.ctx = frame.getElementsByClassName("myChart")[0]
   this.makeScatterData = function () {
      var rs = [];
      var a = getRandomInclusive(MIN, MAX)
      for (var i = 0; i < 100; i++) {
         var x = getRandomInclusive(-1, 1);
         var b = getRandomInclusive(-0.15, 0.15)
         rs.push({
            x: x,
            y: a * x + b,
         });
      }
      return rs;
   }
   this.makeLineData = function (coeffArray) {
      return [{
         x: -1.02, y: - coeffArray[0] * 1.02 + coeffArray[1]
      }, {
         x: 1.02, y: coeffArray[0] * 1.02 + coeffArray[1]
      }];
   }
   this.halt = function () {
      this.unlockInput();
      this.isTraining = false;
   }
   this.iterationNumber = 2000;
   this.learningRate = 0.5;
   this.reset = function () {
      this.unlockInput();
      this.a = tf.scalar(getRandomInclusive(MIN, MAX)).variable();
      this.b = tf.scalar(getRandomInclusive(-0.5, 0.5)).variable();
      this.lineData = this.makeLineData([this.a.dataSync()[0], this.b.dataSync()[0]]);
      this.scatterData = [];
      this.isTraining = false;
      this.iterationNumber = this.iterationNumber;
      this.learningRate = this.learningRate;
      this.currentIterationNumber = 0;
      this.loss = '----';
      this.scatterData = this.makeScatterData()
      this.drawChart();
   }
   this.onIterationNumberChange = function (newValue) {
      this.iterationNumber = newValue;
   }
   this.onLearningRateChange = function (newValue) {
      this.learningRate = newValue;
   }
   this.unlockInput = function () {
      this.learningRateElement.readOnly = false;
      this.totalIterElement.readOnly = false;
   }
   this.lockInput = function () {
      this.learningRateElement.readOnly = true;
      this.totalIterElement.readOnly = true;
   }
   this.drawIndicator = function () {
      this.learningRateElement.value = this.learningRate;
      this.totalIterElement.value = this.iterationNumber;
      this.iterElement.innerHTML = this.currentIterationNumber.toString();
      this.lossElement.innerHTML = this.loss.toString();
      this.co_a_Element.innerHTML = this.a.dataSync()[0].toString();
      this.co_b_Element.innerHTML = this.b.dataSync()[0].toString();
   }
   this.drawChart = function (dontUpdateScatterData) {
      this.drawIndicator();
      if (this.ctx.myChart == null) {
         this.ctx.myChart = new Chart.Scatter(this.ctx, {
            data: {
               datasets: [{
                  type: 'scatter',
                  label: 'Dataset',
                  backgroundColor: 'rgb(12, 40, 86)',
                  data: this.scatterData,
                  showLine: false
               }, {
                  type: 'line', lineTension: 0,
                  label: 'Interpolated Line',
                  backgroundColor: 'rgb(255, 0, 0)',
                  borderColor: 'rgb(255, 0, 0)',
                  data: this.lineData,
                  pointRadius: 0,
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
                        min: -1, max: 1
                     }
                  }],
                  yAxes: [{
                     type: 'linear',
                     ticks: {
                        min: -1, max: 1
                     }
                  }]
               }
            }
         });
      }
      else {
         dontUpdateScatterData !== true && (this.ctx.myChart.data.datasets[0].data = this.scatterData);
         this.ctx.myChart.data.datasets[1].data = this.lineData;
         this.ctx.myChart.update();
      }
   }
   this.train = function () {
      if (this.isTraining) return;
      this.isTraining = true;
      this.lockInput();
      const optimizer = tf.train.sgd(this.learningRate)
      function predict(x, a, b) {
         return tf.tidy(() => {
            return a.mul(x).add(b);
         });
      }
      function calculateLoss(pred, labels) {
         return pred.sub(labels).square().mean();
      }
      function _train(x, y, a, b, _iterationNumber, _currentIterationNumber) {
         var i = _currentIterationNumber;
         if (this.isTraining == false) return;
         if (i >= _iterationNumber) {
            this.isTraining = false;
            this.unlockInput();
            return;
         }
         this.currentIterationNumber = i + 1;
         this.loss = optimizer.minimize(function () {
            var stepLoss = calculateLoss(predict(x, a, b), y);
            return stepLoss;
         }, true).dataSync()[0];
         this.lineData = this.makeLineData([a.dataSync()[0], b.dataSync()[0]]);
         this.drawChart(true);
         window.requestAnimationFrame((function () {
            _train.call(this, x, y, a, b, _iterationNumber, i + 1);
         }).bind(this));
      }
      var xVector = [];
      var yVector = [];
      for (var i = 0; i < this.scatterData.length; i++) {
         xVector.push(this.scatterData[i].x);
         yVector.push(this.scatterData[i].y);
      }
      xVector = tf.tensor1d(xVector);
      yVector = tf.tensor1d(yVector);
      _train.call(this, xVector, yVector, this.a, this.b, this.iterationNumber, 0);
   }
}