document.addEventListener('DOMContentLoaded', function() {
  var container = document.getElementById('ppo-clipping-explorer');
  if (!container) return;

  container.className = 'interactive-container';
  container.innerHTML = `
    <div class="interactive-title">PPO Clipping Explorer</div>
    <div style="font-size:12px; margin-bottom:12px; color:#555;">
      Drag the probability ratio slider to observe how PPO's clipped surrogate loss handles policy updates.
      We assume a clip parameter <strong style="color:#6e46be;">ε = 0.20</strong> (clipping boundary [0.80, 1.20]).
    </div>

    <!-- Configuration slider -->
    <div style="margin-bottom:15px; background:#fff; border:1px solid #eee; padding:12px; border-radius:6px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <label style="font-size:12px; font-weight:600;">
          Probability Ratio r_t(θ): <span id="ratio-val" style="color:#6e46be; font-size:14px;">1.00</span>
        </label>
        <span style="font-size:10px; color:#999;">r_t = π_θ(a|s) / π_old(a|s)</span>
      </div>
      <input type="range" id="ratio-slider" min="0.4" max="1.6" step="0.01" value="1.00" style="width:100%;">
      <div style="display:flex; justify-content:space-between; font-size:9px; color:#999; margin-top:2px;">
        <span>0.40 (Action rare now)</span>
        <span>1.00 (Unchanged)</span>
        <span>1.60 (Action common now)</span>
      </div>
    </div>

    <!-- Side-by-side plots using simple SVGs -->
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-bottom:15px;">
      <!-- Plot A > 0 -->
      <div style="background:#fff; border:1px solid #eee; padding:10px; border-radius:6px; text-align:center;">
        <div style="font-size:11px; font-weight:600; color:#2e7d32; margin-bottom:8px;">Positive Advantage (A = +1.0)</div>
        <svg id="svg-pos" viewBox="0 0 200 150" style="width:100%; height:130px; display:block; margin:0 auto;"></svg>
        <div style="font-size:10px; margin-top:5px; color:#555; background:#f9f9f9; padding:5px; border-radius:4px;" id="calc-pos"></div>
      </div>

      <!-- Plot A < 0 -->
      <div style="background:#fff; border:1px solid #eee; padding:10px; border-radius:6px; text-align:center;">
        <div style="font-size:11px; font-weight:600; color:#d32f2f; margin-bottom:8px;">Negative Advantage (A = -1.0)</div>
        <svg id="svg-neg" viewBox="0 0 200 150" style="width:100%; height:130px; display:block; margin:0 auto;"></svg>
        <div style="font-size:10px; margin-top:5px; color:#555; background:#f9f9f9; padding:5px; border-radius:4px;" id="calc-neg"></div>
      </div>
    </div>
  `;

  var slider = document.getElementById('ratio-slider');
  var valLabel = document.getElementById('ratio-val');
  var svgPos = document.getElementById('svg-pos');
  var svgNeg = document.getElementById('svg-neg');
  var calcPos = document.getElementById('calc-pos');
  var calcNeg = document.getElementById('calc-neg');

  var eps = 0.2;

  slider.addEventListener('input', update);

  function update() {
    var r = parseFloat(slider.value);
    valLabel.textContent = r.toFixed(2);

    drawGraph(svgPos, r, 1.0, calcPos, '#2e7d32');
    drawGraph(svgNeg, r, -1.0, calcNeg, '#d32f2f');
  }

  function drawGraph(svg, r, A, calcBox, activeColor) {
    svg.innerHTML = '';
    var w = 200;
    var h = 150;
    
    // Draw axes
    // X axis y=100. Y axis x=20
    var xAxisY = A > 0 ? 120 : 30;
    var originX = 20;

    var lineX = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lineX.setAttribute('x1', originX);
    lineX.setAttribute('y1', xAxisY);
    lineX.setAttribute('x2', w - 10);
    lineX.setAttribute('y2', xAxisY);
    lineX.setAttribute('stroke', '#ccc');
    svg.appendChild(lineX);

    var lineY = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lineY.setAttribute('x1', originX);
    lineY.setAttribute('y1', 10);
    lineY.setAttribute('x2', originX);
    lineY.setAttribute('y2', h - 10);
    lineY.setAttribute('stroke', '#ccc');
    svg.appendChild(lineY);

    // Map r_t from [0.4, 1.6] to [20, 190]
    function getX(val) {
      return originX + ((val - 0.4) / 1.2) * (w - originX - 20);
    }
    
    // Map Loss from [-1.5, 1.5] to [h-10, 10]
    function getY(loss) {
      var minL = A > 0 ? -0.2 : -1.5;
      var maxL = A > 0 ? 1.5 : 0.2;
      return h - 15 - ((loss - minL) / (maxL - minL)) * (h - 30);
    }

    // Draw grid bounds for 0.8 and 1.2
    var gridL = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    gridL.setAttribute('x1', getX(0.8));
    gridL.setAttribute('y1', 10);
    gridL.setAttribute('x2', getX(0.8));
    gridL.setAttribute('y2', h - 10);
    gridL.setAttribute('stroke', '#eee');
    gridL.setAttribute('stroke-dasharray', '2,2');
    svg.appendChild(gridL);

    var gridR = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    gridR.setAttribute('x1', getX(1.2));
    gridR.setAttribute('y1', 10);
    gridR.setAttribute('x2', getX(1.2));
    gridR.setAttribute('y2', h - 10);
    gridR.setAttribute('stroke', '#eee');
    gridR.setAttribute('stroke-dasharray', '2,2');
    svg.appendChild(gridR);

    // Draw unclipped dotted line
    var pointsDotted = '';
    var pointsSolid = '';
    
    for (var xVal = 0.4; xVal <= 1.6; xVal += 0.05) {
      var lossUnclipped = xVal * A;
      var r_clipped = Math.min(Math.max(xVal, 1 - eps), 1 + eps);
      var lossClipped = r_clipped * A;
      var lossFinal = A > 0 ? Math.min(lossUnclipped, lossClipped) : Math.min(lossUnclipped, lossClipped); // Actually min in formula is mathematically min
      
      pointsDotted += getX(xVal) + ',' + getY(lossUnclipped) + ' ';
      pointsSolid += getX(xVal) + ',' + getY(lossFinal) + ' ';
    }

    var pathDotted = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    pathDotted.setAttribute('fill', 'none');
    pathDotted.setAttribute('stroke', '#bbb');
    pathDotted.setAttribute('stroke-dasharray', '2,2');
    pathDotted.setAttribute('points', pointsDotted);
    svg.appendChild(pathDotted);

    var pathSolid = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    pathSolid.setAttribute('fill', 'none');
    pathSolid.setAttribute('stroke', activeColor);
    pathSolid.setAttribute('stroke-width', '2.5');
    pathSolid.setAttribute('points', pointsSolid);
    svg.appendChild(pathSolid);

    // Calculate current point values
    var lossUnc = r * A;
    var rClipVal = Math.min(Math.max(r, 1 - eps), 1 + eps);
    var lossClip = rClipVal * A;
    var lossFinalVal = Math.min(lossUnc, lossClip);

    // Draw active point circle
    var activeX = getX(r);
    var activeY = getY(lossFinalVal);

    var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', activeX);
    circle.setAttribute('cy', activeY);
    circle.setAttribute('r', '5');
    circle.setAttribute('fill', activeColor);
    circle.setAttribute('stroke', '#fff');
    circle.setAttribute('stroke-width', '1.5');
    svg.appendChild(circle);

    // Grid labels
    var text1 = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text1.setAttribute('x', getX(0.8));
    text1.setAttribute('y', h - 2);
    text1.setAttribute('font-size', '8px');
    text1.setAttribute('fill', '#999');
    text1.setAttribute('text-anchor', 'middle');
    text1.textContent = '0.8';
    svg.appendChild(text1);

    var text2 = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text2.setAttribute('x', getX(1.2));
    text2.setAttribute('y', h - 2);
    text2.setAttribute('font-size', '8px');
    text2.setAttribute('fill', '#999');
    text2.setAttribute('text-anchor', 'middle');
    text2.textContent = '1.2';
    svg.appendChild(text2);

    // Update dynamic explanation text
    var isClipped = r < 0.8 || r > 1.2;
    var stateWord = isClipped ? '<span style="color:#d32f2f; font-weight:600;">Clipped</span>' : '<span style="color:#2e7d32; font-weight:600;">Active</span>';
    calcBox.innerHTML = `
      Unclipped: ${lossUnc.toFixed(2)}<br>
      Clipped (0.8, 1.2): ${lossClip.toFixed(2)}<br>
      <strong>L^CLIP = ${lossFinalVal.toFixed(2)} (${stateWord})</strong>
    `;
  }

  // Initial draw
  update();
});
