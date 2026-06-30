document.addEventListener('DOMContentLoaded', function() {
  var container = document.getElementById('grpo-memory-calculator');
  if (!container) return;

  container.className = 'interactive-container';
  container.innerHTML = `
    <div class="interactive-title">GRPO vs PPO Memory Calculator</div>
    <div style="font-size:12px; margin-bottom:12px; color:#555;">
      Compare the GPU memory footprint of training an LLM using **PPO** (requires Critic + Reward + Ref + Actor)
      versus **GRPO** (requires only Ref + Actor).
    </div>

    <!-- Inputs -->
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-bottom:15px; background:#fff; border:1px solid #eee; padding:12px; border-radius:6px;">
      <div>
        <label style="font-size:12px; font-weight:600; display:block; margin-bottom:5px;">Model Parameter Size:</label>
        <select id="model-size" style="width:100%; padding:6px; font-size:12px; border-radius:4px; border:1px solid #ccc;">
          <option value="7">7 Billion (7B)</option>
          <option value="14" selected>14 Billion (14B - e.g. Qwen)</option>
          <option value="32">32 Billion (32B)</option>
          <option value="70">70 Billion (70B - Llama-3)</option>
        </select>
      </div>
      <div>
        <label style="font-size:12px; font-weight:600; display:block; margin-bottom:5px;">Precision Format:</label>
        <select id="precision" style="width:100%; padding:6px; font-size:12px; border-radius:4px; border:1px solid #ccc;">
          <option value="2">Half-Precision (bfloat16 / 16-bit) - 2 bytes/param</option>
          <option value="4">Full-Precision (32-bit) - 4 bytes/param</option>
        </select>
      </div>
    </div>

    <!-- Visual Output Comparison -->
    <div style="background:#fff; border:1px solid #eee; padding:15px; border-radius:6px; margin-bottom:15px;">
      <!-- PPO Bar -->
      <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; font-size:11px; font-weight:600; margin-bottom:4px;">
          <span>PPO Footprint (4 Models)</span>
          <span id="ppo-mem-text" style="color:#e53935;">- GB</span>
        </div>
        <div style="width:100%; background:#f1f5f9; height:12px; border-radius:6px; overflow:hidden;">
          <div id="ppo-bar" style="background:#e53935; height:100%; width:0%; transition:width 0.4s ease;"></div>
        </div>
      </div>

      <!-- GRPO Bar -->
      <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; font-size:11px; font-weight:600; margin-bottom:4px;">
          <span>GRPO Footprint (2 Models)</span>
          <span id="grpo-mem-text" style="color:#2e7d32;">- GB</span>
        </div>
        <div style="width:100%; background:#f1f5f9; height:12px; border-radius:6px; overflow:hidden;">
          <div id="grpo-bar" style="background:#2e7d32; height:100%; width:0%; transition:width 0.4s ease;"></div>
        </div>
      </div>
    </div>

    <!-- Explanation Box -->
    <div id="calculator-explanation" style="font-size:11px; color:#555; background:#f8fafc; padding:10px; border-radius:6px; border-left:3px solid #6e46be; line-height:1.5;">
      <!-- Dynamic explanation content -->
    </div>
  `;

  // DOM elements
  var modelSizeSelect = document.getElementById('model-size');
  var precisionSelect = document.getElementById('precision');
  var ppoMemText = document.getElementById('ppo-mem-text');
  var grpoMemText = document.getElementById('grpo-mem-text');
  var ppoBar = document.getElementById('ppo-bar');
  var grpoBar = document.getElementById('grpo-bar');
  var explanation = document.getElementById('calculator-explanation');

  // Change listeners
  modelSizeSelect.addEventListener('change', calculate);
  precisionSelect.addEventListener('change', calculate);

  function calculate() {
    var size = parseFloat(modelSizeSelect.value);      // e.g. 14 for 14B
    var bytes = parseFloat(precisionSelect.value);     // 2 or 4

    // Formula rules:
    // Memory per parameter during training (including Optimizer states & gradients):
    // Standard training parameters footprint (AdamW):
    // - Model weights: B bytes
    // - Gradients: B bytes
    // - Optimizer states (AdamW): 8 bytes
    // Total Trainable model multiplier: B + B + 8 = 2B + 8 bytes/param.
    // E.g. bf16 (2 bytes): 2(2) + 8 = 12 bytes/param.
    // E.g. fp32 (4 bytes): 2(4) + 8 = 16 bytes/param.
    var multiplierTrainable = 2 * bytes + 8;

    // Frozen reference / reward models:
    // Only requires model weights in memory (no optimizer states or gradients):
    // Total Frozen model multiplier: B bytes/param.
    var multiplierFrozen = bytes;

    // PPO components:
    // 1. Trainable Actor: (2B + 8) * Size
    // 2. Trainable Critic: (2B + 8) * Size (usually same size as actor)
    // 3. Frozen Reference Model: B * Size
    // 4. Frozen Reward Model: B * Size
    var ppoMem = (multiplierTrainable * size) + (multiplierTrainable * size) + (multiplierFrozen * size) + (multiplierFrozen * size);

    // GRPO components:
    // 1. Trainable Actor: (2B + 8) * Size
    // 2. Frozen Reference Model: B * Size
    // (No Critic and No separate Reward model during training step, standard setup)
    var grpoMem = (multiplierTrainable * size) + (multiplierFrozen * size);

    // Display
    ppoMemText.textContent = ppoMem.toFixed(0) + ' GB';
    grpoMemText.textContent = grpoMem.toFixed(0) + ' GB';

    // Scale bars: max height 100% represents 2000 GB
    var maxDisplay = 2000;
    var ppoPct = Math.min((ppoMem / maxDisplay) * 100, 100);
    var grpoPct = Math.min((grpoMem / maxDisplay) * 100, 100);

    ppoBar.style.width = ppoPct + '%';
    grpoBar.style.width = grpoPct + '%';

    var savedGb = ppoMem - grpoMem;
    var savedPct = ((ppoMem - grpoMem) / ppoMem * 100).toFixed(0);

    explanation.innerHTML = `
      <strong>Memory Savings: ${savedGb.toFixed(0)} GB (${savedPct}% reduction)</strong><br>
      • **PPO components:** Actor (${(multiplierTrainable * size).toFixed(0)} GB) + Critic (${(multiplierTrainable * size).toFixed(0)} GB) + Reference (${(multiplierFrozen * size).toFixed(0)} GB) + Reward (${(multiplierFrozen * size).toFixed(0)} GB).<br>
      • **GRPO components:** Actor (${(multiplierTrainable * size).toFixed(0)} GB) + Reference (${(multiplierFrozen * size).toFixed(0)} GB) only. The Critic network is completely removed, freeing up massive GPU memory for larger context window or batch sizes.
    `;
  }

  // Initial calculation
  calculate();
});
