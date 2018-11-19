function getRandomInclusive(min, max) {
   var r = Math.random();
   return (r >= 0.5 ? 1.5 - r : r) * (max - min) + min;
}
