document.addEventListener('DOMContentLoaded', function() {
  var container = document.getElementById('bandit-simulator');
  if (!container) return;

  // Add styles and structure
  container.className = 'interactive-container';
  container.innerHTML = `
    <div class="interactive-title">ε-Greedy Bandit Simulator</div>
    <div style="font-size:12px; margin-bottom:12px; color:#555;">
      Understand how the exploration parameter ε affects learning. We have 3 slot machine arms with true hidden mean rewards:
      <strong style="color:#2e7d32">Arm 1 = 1.5</strong>, <strong style="color:#d97706">Arm 2 = 0.8</strong>, <strong style="color:#6e46be">Arm 3 = 2.2</strong> (the best arm).
    </div>
    
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:15px;">
      <div>
        <label style="font-size:12px; font-weight:600; display:block; margin-bottom:5px;">
          Exploration Rate (ε): <span id="eps-val" style="color:#6e46be;">0.10</span>
        </label>
        <input type="range" id="eps-slider" min="0" max="0.5" step="0.01" value="0.10" style="width:100%;">
        <div style="display:flex; justify-content:space-between; font-size:10px; color:#999; margin-top:2px;">
          <span>0.00 (Greedy)</span>
          <span>0.10</span>
          <span>0.50</span>
        </div>
      </div>
      <div style="display:flex; align-items:flex-end; gap:8px;">
        <button id="btn-pull" style="padding:8px 16px; background:#6e46be; color:#fff; border:none; border-radius:6px; font-size:11px; cursor:pointer; font-weight:600; transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(110, 70, 190, 0.2);">Pull 1 Step</button>
        <button id="btn-run-100" style="padding:8px 16px; background:#ffffff; border:1px solid #d1d5db; border-radius:6px; font-size:11px; cursor:pointer; font-weight:600; color:#374151; transition: all 0.2s ease; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);">Run 100 Steps</button>
        <button id="btn-reset" style="padding:8px 12px; background:#ffffff; border:1px solid #d1d5db; border-radius:6px; font-size:11px; cursor:pointer; color:#9ca3af; transition: all 0.2s ease;">Reset</button>
      </div>
    </div>

    <!-- Active Display Grid -->
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
      <!-- Stats Table -->
      <div style="background:#fff; border:1px solid #eee; padding:12px; border-radius:6px;">
        <div style="font-size:11px; font-weight:600; color:#6e46be; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;">Estimated Action Values</div>
        <table style="width:100%; font-size:11px; border-collapse:collapse;" id="stats-table">
          <thead>
            <tr style="border-bottom:1px solid #eee; text-align:left; color:#666;">
              <th style="padding:4px 0;">Arm</th>
              <th style="padding:4px 0;">Hidden Mean</th>
              <th style="padding:4px 0;">Pulls</th>
              <th style="padding:4px 0;">Q(a) (Estimate)</th>
            </tr>
          </thead>
          <tbody>
            <!-- Dynamic rows -->
          </tbody>
        </table>
        <div style="margin-top:10px; font-size:10px; color:#666;" id="step-info">
          Total steps: 0 | Selected: -
        </div>
      </div>

      <!-- Live Line Chart -->
      <div style="background:#fff; border:1px solid #eee; padding:12px; border-radius:6px; text-align:center;">
        <div style="font-size:11px; font-weight:600; color:#6e46be; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;">Average Reward History</div>
        <svg id="chart-svg" viewBox="0 0 240 120" style="width:100%; height:95px; display:block; margin:0 auto;"></svg>
      </div>
    </div>
  `;

  // Bandit properties
  var means = [1.5, 0.8, 2.2];
  var pulls = [0, 0, 0];
  var Q = [0, 0, 0];
  var totalSteps = 0;
  var averageRewardHistory = [];
  var rewardSum = 0;
  var lastPullInfo = "";

  // DOM references
  var slider = document.getElementById('eps-slider');
  var epsVal = document.getElementById('eps-val');
  var btnPull = document.getElementById('btn-pull');
  var btnRun100 = document.getElementById('btn-run-100');
  var btnReset = document.getElementById('btn-reset');
  var statsTable = document.getElementById('stats-table').querySelector('tbody');
  var stepInfo = document.getElementById('step-info');
  var chartSvg = document.getElementById('chart-svg');

  // Add CSS effects to buttons
  var style = document.createElement('style');
  style.innerHTML = `
    #btn-pull:hover { background: #5a37a3 !important; transform: translateY(-1px); box-shadow: 0 4px 6px rgba(110, 70, 190, 0.3) !important; }
    #btn-run-100:hover { background: #f9fafb !important; border-color: #9ca3af !important; color: #111827 !important; transform: translateY(-1px); }
    #btn-reset:hover { background: #f9fafb !important; color: #ef4444 !important; border-color: #fca5a5 !important; }
  `;
  document.head.appendChild(style);

  // Listeners
  slider.addEventListener('input', function() {
    epsVal.textContent = parseFloat(slider.value).toFixed(2);
  });

  btnPull.addEventListener('click', function() {
    pull(1);
  });

  btnRun100.addEventListener('click', function() {
    pull(100);
  });

  btnReset.addEventListener('click', function() {
    pulls = [0, 0, 0];
    Q = [0, 0, 0];
    totalSteps = 0;
    averageRewardHistory = [];
    rewardSum = 0;
    lastPullInfo = "Simulator Reset.";
    updateUI();
  });

  function pull(steps) {
    var eps = parseFloat(slider.value);
    
    for (var s = 0; s < steps; s++) {
      var arm;
      
      // Epsilon-greedy check
      if (Math.random() < eps) {
        // Explore
        arm = Math.floor(Math.random() * 3);
      } else {
        // Exploit (break ties randomly)
        var maxVal = Math.max.apply(null, Q);
        var candidates = [];
        for (var i = 0; i < 3; i++) {
          if (Q[i] === maxVal) candidates.push(i);
        }
        arm = candidates[Math.floor(Math.random() * candidates.length)];
      }

      // Generate reward under normal distribution (simplified)
      var reward = means[arm] + (Math.random() + Math.random() + Math.random() - 1.5); // ~ N(mean, 0.25) approx
      pulls[arm]++;
      
      // Update Q value incrementally: Q_{n+1} = Q_n + 1/n * (R_n - Q_n)
      Q[arm] = Q[arm] + (1 / pulls[arm]) * (reward - Q[arm]);
      
      totalSteps++;
      rewardSum += reward;
      averageRewardHistory.push(rewardSum / totalSteps);
      
      lastPullInfo = "Pulled Arm " + (arm + 1) + " (Reward: " + reward.toFixed(2) + ")";
    }
    
    updateUI();
  }

  function updateUI() {
    // 1. Table rows
    var html = '';
    for (var i = 0; i < 3; i++) {
      var armColor = i === 0 ? '#2e7d32' : (i === 1 ? '#d97706' : '#6e46be');
      html += `
        <tr style="border-bottom:1px solid #f9f9f9;">
          <td style="padding:6px 0; font-weight:600; color:${armColor}">Arm \${i+1}</td>
          <td style="padding:6px 0; color:#555;">\${means[i].toFixed(1)}</td>
          <td style="padding:6px 0; color:#555;">\${pulls[i]}</td>
          <td style="padding:6px 0; font-weight:700; color:#111;">\${Q[i].toFixed(3)}</td>
        </tr>
      `;
    }
    statsTable.innerHTML = html;

    // 2. Info string
    stepInfo.textContent = "Total pulls: " + totalSteps + " | " + lastPullInfo;

    // 3. Render chart
    drawChart();
  }

  function drawChart() {
    chartSvg.innerHTML = '';
    var w = 240;
    var h = 120;
    var padding = 20;

    // Draw axes
    var lineX = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lineX.setAttribute('x1', padding);
    lineX.setAttribute('y1', h - padding);
    lineX.setAttribute('x2', w - 10);
    lineX.setAttribute('y2', h - padding);
    lineX.setAttribute('stroke', '#ccc');
    chartSvg.appendChild(lineX);

    var lineY = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    lineY.setAttribute('x1', padding);
    lineY.setAttribute('y1', 10);
    lineY.setAttribute('x2', padding);
    lineY.setAttribute('y2', h - padding);
    lineY.setAttribute('stroke', '#ccc');
    chartSvg.appendChild(lineY);

    if (averageRewardHistory.length === 0) {
      // Draw placeholder text
      var txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      txt.setAttribute('x', w / 2 + 10);
      txt.setAttribute('y', h / 2);
      txt.setAttribute('font-size', '10px');
      txt.setAttribute('fill', '#999');
      txt.setAttribute('text-anchor', 'middle');
      txt.textContent = "Pull arms to see learning trajectory";
      chartSvg.appendChild(txt);
      return;
    }

    // Determine min/max values
    var minVal = 0;
    var maxVal = 2.5;

    // Map history points to coordinates
    var pts = '';
    var skip = Math.max(1, Math.floor(averageRewardHistory.length / 100)); // limit points to 100 max
    
    for (var i = 0; i < averageRewardHistory.length; i += skip) {
      var cx = padding + (i / (averageRewardHistory.length - 1 || 1)) * (w - padding - 15);
      var val = averageRewardHistory[i];
      // Bound
      var cy = h - padding - ((val - minVal) / (maxVal - minVal)) * (h - padding - 20);
      pts += cx + ',' + cy + ' ';
    }

    // Polyline
    var poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    poly.setAttribute('fill', 'none');
    poly.setAttribute('stroke', '#6e46be');
    poly.setAttribute('stroke-width', '2');
    poly.setAttribute('points', pts);
    chartSvg.appendChild(poly);

    // Draw reference horizontal dashed line at 2.2 (best value)
    var targetY = h - padding - ((2.2 - minVal) / (maxVal - minVal)) * (h - padding - 20);
    var refLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    refLine.setAttribute('x1', padding);
    refLine.setAttribute('y1', targetY);
    refLine.setAttribute('x2', w - 10);
    refLine.setAttribute('y2', targetY);
    refLine.setAttribute('stroke', '#2e7d32');
    refLine.setAttribute('stroke-width', '1');
    refLine.setAttribute('stroke-dasharray', '3,3');
    chartSvg.appendChild(refLine);

    // Add labels on Y axis
    var grid = [0.0, 1.0, 2.0, 2.2];
    for (var gIdx = 0; gIdx < grid.length; gIdx++) {
      var g = grid[gIdx];
      var gy = h - padding - ((g - minVal) / (maxVal - minVal)) * (h - padding - 20);
      
      var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', padding);
      line.setAttribute('y1', gy);
      line.setAttribute('x2', w - padding);
      line.setAttribute('y2', gy);
      line.setAttribute('stroke', '#eee');
      line.setAttribute('stroke-dasharray', '2,2');
      chartSvg.appendChild(line);

      var txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      txt.setAttribute('x', padding - 2);
      txt.setAttribute('y', gy + 3);
      txt.setAttribute('font-size', '8px');
      txt.setAttribute('fill', '#999');
      txt.setAttribute('text-anchor', 'end');
      txt.textContent = g.toFixed(1);
      chartSvg.appendChild(txt);
    }
  }

  // Initial draw
  updateUI();
});
