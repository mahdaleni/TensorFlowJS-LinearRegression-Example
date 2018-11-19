function PolynomialRegression(frame) {
   this.learningRateElement = frame.getElementsByClassName('learningRate')[0]
   this.iterElement = frame.getElementsByClassName('iter')[0]
   this.totalIterElement = frame.getElementsByClassName('totalIter')[0]
   this.lossElement = frame.getElementsByClassName('loss')[0]
   this.co_a_Element = frame.getElementsByClassName('co_a')[0]
   this.co_b_Element = frame.getElementsByClassName('co_b')[0]
   this.co_c_Element = frame.getElementsByClassName('co_c')[0]
   this.co_d_Element = frame.getElementsByClassName('co_d')[0]
   this.ctx = frame.getElementsByClassName("myChart")[0]
   this.makeScatterData = function () {
      var rs = [];
      var a = -0.8;
      var b = -0.2;
      var c = 0.9;
      var d = 0.5;
      for (var i = 0; i < 100; i++) {
         var x = getRandomInclusive(-1, 1)
         rs.push({
            x: x,
            y: a * Math.pow(x, 3) +
               b * Math.pow(x, 2) +
               c * x +
               d + getRandomInclusive(-0.05, 0.05)
         });
      }
      return rs;
   }
   this.makeLineData = function (coeffArray) {
      var rs = [];
      var a = coeffArray[0], b = coeffArray[1], c = coeffArray[2], d = coeffArray[3];
      for (var _x = -1.02; _x <= 1.02; _x += 0.02) {
         rs.push({
            x: _x,
            y: a * Math.pow(_x, 3) +
               b * Math.pow(_x, 2)  +
               c * _x +
               d,
         });
      }
      return rs;
   }
   this.halt = function () {
      this.unlockInput();
      this.isTraining = false;
   }
   this.iterationNumber = 2000;
   this.learningRate = 0.5;
   this.reset = function () {
      this.unlockInput();
      this.a = tf.scalar(Math.random()).variable();
      this.b = tf.scalar(Math.random()).variable();
      this.c = tf.scalar(Math.random()).variable();
      this.d = tf.scalar(Math.random()).variable();
      this.lineData = this.makeLineData([
         this.a.dataSync()[0],
         this.b.dataSync()[0],
         this.c.dataSync()[0],
         this.d.dataSync()[0],
      ]);
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
      this.co_c_Element.innerHTML = this.c.dataSync()[0].toString();
      this.co_d_Element.innerHTML = this.d.dataSync()[0].toString();
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
                     ticks: {
                        min: -1, max: 1
                     }
                  }],
                  yAxes: [{
                     type: 'linear',
                     ticks: {
                        min: 0, max: 1
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
      function predict(x, a, b, c, d) {
         return tf.tidy(() => {
            return a.mul(x.pow(tf.scalar(3, 'int32')))
               .add(b.mul(x.square()))
               .add(c.mul(x))
               .add(d);
         });
      }
      function calculateLoss(pred, labels) {
         return pred.sub(labels).square().mean();
      }
      function _train(x, y, a, b, c, d, _iterationNumber, _currentIterationNumber) {
         var i = _currentIterationNumber;
         if (this.isTraining == false) return;
         if (i >= _iterationNumber) {
            this.isTraining = false;
            this.unlockInput();
            return;
         }
         this.currentIterationNumber = i + 1;
         this.loss = optimizer.minimize(function () {
            var stepLoss = calculateLoss(predict(x, a, b, c, d), y);
            return stepLoss;
         }, true).dataSync()[0];
         this.lineData = this.makeLineData([
            a.dataSync()[0],
            b.dataSync()[0],
            c.dataSync()[0],
            d.dataSync()[0],
         ]);
         this.drawChart(true);
         window.requestAnimationFrame((function () {
            _train.call(this, x, y, a, b, c, d, _iterationNumber, i + 1);
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
      _train.call(this, xVector, yVector, this.a, this.b, this.c, this.d, this.iterationNumber, 0);
   }
}