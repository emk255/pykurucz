/* Adapted from https://github.com/tingyuansen/bob-portrait
 * MIT-licensed by Yuan-Sen Ting; embedded here so it shares our paper-tone
 * background and renders inline (no iframe). The three on-page controls
 * (Explode / Mono / Palette) and keyboard shortcuts have been removed —
 * mouse hover is the only interaction. */

(function () {
  var canvas = document.getElementById('pk-bob-canvas');
  if (!canvas) return; // only the homepage hero has this canvas

  // ── ASCII art ─────────────────────────────────────────────────────────────
  var PORTRAIT_RAW = [
    "                                            ....",
    "                              ......:::::::=*#%*-:.",
    "                        ..:=+##**+=+#%@@@%@%%%%%@%*.",
    "                       .=+#@@@@@%%@@@@%%@@@@@%%%%%%%#*-",
    "                     .-*#%%%%%%%%%%%%%@@%%%%@@@@%%%%%%%*--:",
    "                   .=#@@@%%%%%@@@@@@@@@@@@@@@@@@@%%%%%%@@@#.",
    "                  -%@@%%%@%%@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%*=:",
    "                 -%@%%%%%%%@@@%%%%%%%%%%%@@@@@%@@%%%%%%%%%#%@*.",
    "               :+%%%%%%%%%%%%%@@@@@@@@@@@@@%%@@%%%##%%%%%%%%%#%+",
    "              =#%%@%%%@%%%@@@@%##****##%%%%@@%%%#####%%%%@@@%#%@*",
    "             =%#%%%%@%*=+**+=-:..::-----=++**++++++++*#%%%%%%%%%*:",
    "            :%%@@@@@%*-...      ..::--:::::----===++++++#%@@%%%%**-",
    "            #%%%%%%%#=-:..  ..    ...::--::---==++**###*+#%%%@@%%#%-",
    "            +##%#%%%*=:..  .--::::::::-:--:---+**********++**#%@%##+",
    "          .+*##%%%%%*-::.:::==----=====::-===+*+++*****###=%@%%%@%%#=.",
    "         ::+#%%%%%@%#=-:-:::---=+*++=-::.:=*+*####%@###*##-*%@@@%@@##+-",
    "           :#%%%%%%%%#+=:::.:*++%%*++-:..:-+##%%#******+*+=**%%%%%%%##%=",
    "           .=#%##%%%%*:::.. ....:---=:....:=+**###******+=+==*%%#%%%%:.",
    "             #**#%%%%*:.. .     .:::......::--**+******++**+=+%#+##*%",
    "             -*+#**+#*-..   .....::....  ..-=-=##***+**####*==+++*%#+",
    "              :++---++-:..    .   . .:.:-::=#%#+%@%#*++=*##*=-=*##@%*",
    "                -..--:=:....... .  .=:.+#--+*%@%%%@%***==##*+-=+#*#%+",
    "                  .-: .::........ .=: .:--=+=+*#*#%%%##*=*#+=-=#*#+-",
    "                  ..::..:--.  ....-=:-==+*********#%%%%*+*#=--==--.",
    "                  ..  .::-=:......-=+**###*##*#%%%%%%%#****+-:.",
    "                     ..:::-:::::.:=+##*++===+*##%%%%%%#***+*=:",
    "                          ----:-:-=*%+::::-=+++*##%%%%###***=.",
    "                     .:-++::::-:-=+*%=--=++*++**#%%%@%%###*=*%#*=-:.",
    "               .-=+*#%%@@@#+=---==**#**+==++++**#%%%%##**++:.#@@@@%%#+=-.",
    "         .:=+*###%%%%%@%%%@#.:-==-=+++**+=++***##%##*#*+==#- =%%%%@@%%%%%#*+-.",
    "   .:-=+*########%%%%%%%%%*.  .:----=+***##**#######***++%*::*%%%%%@@%%%%%%%%%#*+-.",
    "=+*#########%%%%%%%%%%%%##*.   ..:::-==+***#**#%#**++#%%%#-:-%%%%@@@@%@@@@%%%%%%@@%#*+=:.",
    "#######%%%%%%%%%%%%%%%%##*%=    .:::::---==***+*#*++#%%%#+--#@%%@@@@@@@@%%@@@@%%%%%%%%%%#*",
    "##%%%%%%%%%%%%%%%%%%%%####%.     .::::::---=+**###*%%%%##*-*@%@@@@@@@@@@@@@@%@@@@@@%%%%%%%",
    "%%%%%%%%%%%%%%%%%%#%%####%#        .:-------+####%%%%###*=*@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@%",
    "%%%%%%%%%%%%%%%%%##########.         .:--=--#%%%%%##****+*@%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
    "%%%%%%%%%%%%%%%%#########*=              :=+##*=+++===++*%%%%%@@@%@@@@@@@@@@@@@@@@@@@@@@@@",
    "%%%%%%%%%%%%%%%###########: .           :*%%#%#=::::--+*#%%#*%@%%%@@@@@@@@@@@@@@@@@@@@@@@@",
    "%%%%%%%%%%%%%%############. -         :*@@%@@%@@%*-:-+*####+*@%%%%@@@@@@@@@@@@@@@@@@@@@@%%",
    "%%%%%%%%%%%%%############%- ::        #@@%%%%%%%@@%*+*###*+-#%%%%%@@@@%%@@@@@@@@@@@@@@%%%%",
    "%%%%%%%%%%%%%############%+ :.      .+--%%%@@@@@%%#+*+=---%%%%%%%@@@@@%%%@@@@@@@@@@@@%%%%%"
  ];
  var PORTRAIT_LINES = PORTRAIT_RAW;
  var ROWS = PORTRAIT_LINES.length;
  var COLS = 0;
  for (var li = 0; li < PORTRAIT_LINES.length; li++) {
    if (PORTRAIT_LINES[li].length > COLS) COLS = PORTRAIT_LINES[li].length;
  }

  // ── Config ────────────────────────────────────────────────────────────────
  var FONT_SIZE = 13;
  var LINE_HEIGHT = 15;
  var MONO_FONT = FONT_SIZE + 'px "JetBrains Mono", "Courier New", Courier, monospace';
  var PADDING = 36;
  var SPRING_K = 0.045;
  var DAMPING = 0.88;
  var REPULSION_RADIUS = 90;
  var REPULSION_STRENGTH = 8;
  var GLOW_RADIUS = 140;
  var WAVE_SPEED = 1.8;

  // Density ramp (how "filled" each glyph looks). Drives both alpha & shade.
  var DENSITY_MAP = {
    '.': 0.06, ':': 0.12, '-': 0.15, ',': 0.08, ';': 0.14,
    '=': 0.25, '+': 0.35, '/': 0.20, '\\': 0.20, '|': 0.22,
    '(': 0.28, ')': 0.28, '!': 0.30, '?': 0.32, '^': 0.18,
    '*': 0.50, '#': 0.65, '%': 0.80, '@': 1.0
  };

  // ── Paper palette ─────────────────────────────────────────────────────────
  // Dark ink on light paper, with a navy hover glow that matches our --pk-accent.
  // Low density = faint warm grey (close to paper); high density = ink.
  var PALETTE = {
    lo: [180, 180, 175],   // faint paper grey for sparse chars
    hi: [20, 22, 26],      // dark ink for dense chars
    glow: [70, 90, 160]    // navy hover glow
  };

  function lerpColor(lo, hi, t) {
    return [
      lo[0] + (hi[0] - lo[0]) * t,
      lo[1] + (hi[1] - lo[1]) * t,
      lo[2] + (hi[2] - lo[2]) * t
    ];
  }

  // ── Measure character widths via canvas measureText ───────────────────────
  function measureCharWidths(font) {
    var widths = {};
    var off = document.createElement('canvas').getContext('2d');
    off.font = font;
    var seen = {};
    for (var i = 0; i < PORTRAIT_LINES.length; i++) {
      var line = PORTRAIT_LINES[i];
      for (var j = 0; j < line.length; j++) {
        var ch = line[j];
        if (!seen[ch]) {
          widths[ch] = off.measureText(ch).width;
          seen[ch] = 1;
        }
      }
    }
    return widths;
  }

  // ── Canvas setup ──────────────────────────────────────────────────────────
  var ctx = canvas.getContext('2d');
  var dpr = window.devicePixelRatio || 1;

  var particles = [];
  var mouseX = -9999, mouseY = -9999;
  var startTime = 0;
  var logicalW = 0, logicalH = 0;

  function init() {
    var monoWidths = measureCharWidths(MONO_FONT);
    var monoCharW = monoWidths[' '] || 7.8;
    var monoPortraitW = COLS * monoCharW;
    var monoPortraitH = ROWS * LINE_HEIGHT;

    logicalW = monoPortraitW + PADDING * 2;
    logicalH = monoPortraitH + PADDING * 2;

    // Always draw at full natural resolution; the host CSS scales us down to
    // whatever fits in the hero column without distorting glyph proportions.
    canvas.width = Math.round(logicalW * dpr);
    canvas.height = Math.round(logicalH * dpr);
    canvas.style.width = '100%';
    canvas.style.height = 'auto';
    canvas.style.aspectRatio = (logicalW / logicalH).toFixed(4);

    var monoOffsetX = (logicalW - monoPortraitW) / 2;
    var monoOffsetY = (logicalH - monoPortraitH) / 2;
    var centerX = logicalW / 2;
    var centerY = logicalH / 2;

    particles = [];
    for (var row = 0; row < ROWS; row++) {
      var line = PORTRAIT_LINES[row];
      for (var col = 0; col < line.length; col++) {
        var ch = line[col];
        if (ch === ' ') continue;
        var density = DENSITY_MAP[ch] || 0.3;
        var monoX = monoOffsetX + col * monoCharW;
        var monoY = monoOffsetY + row * LINE_HEIGHT;
        var dist = Math.sqrt((monoX - centerX) * (monoX - centerX) + (monoY - centerY) * (monoY - centerY));
        var angle = Math.random() * Math.PI * 2;
        var radius = 200 + Math.random() * 300;
        particles.push({
          char: ch,
          homeX: monoX, homeY: monoY,
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          vx: (Math.random() - 0.5) * 2,
          vy: (Math.random() - 0.5) * 2,
          density: density,
          distFromCenter: dist
        });
      }
    }

    startTime = performance.now();
    requestAnimationFrame(render);
  }

  // ── Mouse mapping (CSS-px → canvas-logical px) ────────────────────────────
  function canvasCoords(clientX, clientY) {
    var rect = canvas.getBoundingClientRect();
    return [
      (clientX - rect.left) * (logicalW / rect.width),
      (clientY - rect.top) * (logicalH / rect.height)
    ];
  }
  canvas.addEventListener('mousemove', function (e) {
    var c = canvasCoords(e.clientX, e.clientY);
    mouseX = c[0]; mouseY = c[1];
  });
  canvas.addEventListener('mouseleave', function () { mouseX = -9999; mouseY = -9999; });
  canvas.addEventListener('touchmove', function (e) {
    if (!e.touches.length) return;
    var t = e.touches[0];
    var c = canvasCoords(t.clientX, t.clientY);
    mouseX = c[0]; mouseY = c[1];
  }, { passive: true });
  canvas.addEventListener('touchend', function () { mouseX = -9999; mouseY = -9999; });

  // ── Render loop ────────────────────────────────────────────────────────────
  function render(now) {
    var elapsed = now - startTime;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, logicalW, logicalH);
    ctx.font = MONO_FONT;
    ctx.textBaseline = 'top';

    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      var activateTime = p.distFromCenter / WAVE_SPEED;
      if (elapsed < activateTime) continue;
      var age = elapsed - activateTime;
      var fadeIn = Math.min(1, age / 600);

      var breatheX = Math.sin(now * 0.0008 + p.homeY * 0.012) * 0.4;
      var breatheY = Math.cos(now * 0.0006 + p.homeX * 0.009) * 0.3;
      var targetX = p.homeX + breatheX;
      var targetY = p.homeY + breatheY;

      // Spring back to home
      p.vx += (targetX - p.x) * SPRING_K;
      p.vy += (targetY - p.y) * SPRING_K;

      // Mouse repulsion
      var mdx = p.x - mouseX, mdy = p.y - mouseY;
      var mDist2 = mdx * mdx + mdy * mdy;
      if (mDist2 < REPULSION_RADIUS * REPULSION_RADIUS && mDist2 > 1) {
        var mDist = Math.sqrt(mDist2);
        var force = REPULSION_STRENGTH * (1 - mDist / REPULSION_RADIUS);
        p.vx += (mdx / mDist) * force;
        p.vy += (mdy / mDist) * force;
      }

      p.vx *= DAMPING;
      p.vy *= DAMPING;
      p.x += p.vx;
      p.y += p.vy;

      // Glow falloff: blends ink toward navy as the cursor approaches.
      var glowDx = p.x - mouseX, glowDy = p.y - mouseY;
      var glowDist = Math.sqrt(glowDx * glowDx + glowDy * glowDy);
      var glow = Math.max(0, 1 - glowDist / GLOW_RADIUS);

      // Base color: lerp from light paper grey → ink by density.
      var t = p.density;
      var base = lerpColor(PALETTE.lo, PALETTE.hi, Math.min(1, 0.15 + t * 0.85));
      // Blend toward glow color when near cursor.
      var c = lerpColor(base, PALETTE.glow, glow * 0.7);
      var alpha = fadeIn * Math.min(1, 0.20 + t * 0.75 + glow * 0.2);

      ctx.fillStyle = 'rgba(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ',' + alpha.toFixed(2) + ')';
      ctx.fillText(p.char, p.x, p.y);
    }

    requestAnimationFrame(render);
  }

  init();
})();
