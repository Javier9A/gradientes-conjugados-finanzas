/* =====================================================================
   Aplicación del Método de Gradientes Conjugados Precondicionado (GCP)
   en la Planificación Financiera · UMSA · Métodos Numéricos
   JavaScript puro, sin dependencias. Abrir index.html en el navegador.
   ===================================================================== */
(function (global) {
  'use strict';

  /* ===================================================================
     1. UTILIDADES DE FORMATO Y LECTURA
     =================================================================== */
  const $ = (id) => document.getElementById(id);
  const SUBD = '₀₁₂₃₄₅₆₇₈₉';
  const SUPM = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const sub = (i) => String(i + 1).replace(/\d/g, (d) => SUBD[+d]);
  const sup = (n) => String(n).replace(/[-0-9]/g, (c) => SUPM[c]);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const NUM_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
  /** Convierte texto en número. Acepta coma o punto decimal. Devuelve NaN si es inválido. */
  function parseNumber(s) {
    if (typeof s === 'number') return Number.isFinite(s) ? s : NaN;
    if (s === null || s === undefined) return NaN;
    const t = String(s).trim().replace('−', '-').replace(',', '.');
    if (!NUM_RE.test(t)) return NaN;
    const v = Number(t);
    return Number.isFinite(v) ? v : NaN;
  }

  function group(intStr) { return intStr.replace(/\B(?=(\d{3})+(?!\d))/g, ' '); }
  function fixed(x, d) {
    const s = Math.abs(x).toFixed(d);
    const parts = s.split('.');
    return (x < 0 && Number(s) !== 0 ? '−' : '') + group(parts[0]) + (parts[1] ? '.' + parts[1] : '');
  }
  function sci(x, d) {
    if (d === undefined) d = 3;
    if (x === null || x === undefined || Number.isNaN(x)) return '—';
    if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
    if (x === 0) return '0';
    const parts = x.toExponential(d).split('e');
    return parts[0].replace('-', '−') + '×10' + sup(parseInt(parts[1], 10));
  }
  function num(x, d) {
    if (d === undefined) d = 4;
    if (x === null || x === undefined || Number.isNaN(x)) return '—';
    if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
    const ax = Math.abs(x);
    if (ax !== 0 && (ax < 1e-4 || ax >= 1e12)) return sci(x, 3);
    return fixed(x, d);
  }

  /* ===================================================================
     2. NÚCLEO NUMÉRICO
     =================================================================== */
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm2(v) {
    let m = 0;
    for (const x of v) m = Math.max(m, Math.abs(x));
    if (m === 0 || !Number.isFinite(m)) return m;
    let s = 0;
    for (const x of v) { const t = x / m; s += t * t; }
    return m * Math.sqrt(s);
  }
  const matVec = (A, x) => A.map((row) => dot(row, x));
  const matClone = (A) => A.map((r) => r.slice());

  /** Factorización de Cholesky A = L·Lᵀ. Falla si algún pivote no es positivo. */
  function cholesky(A) {
    const n = A.length;
    const L = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (!(s > 1e-13 * Math.abs(A[i][i]))) return { ok: false, index: i, pivot: s };
          L[i][i] = Math.sqrt(s);
        } else {
          L[i][j] = s / L[j][j];
        }
      }
    }
    return { ok: true, L };
  }
  /** Resuelve L·Lᵀ x = b por sustitución progresiva y regresiva. */
  function choleskySolve(L, b) {
    const n = L.length;
    const y = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }
  /** Eliminación gaussiana con pivoteo parcial (solo se usa como verificación independiente en QA). */
  function gaussSolve(A, b) {
    const n = A.length;
    const M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (M[p][c] === 0) return null;
      if (p !== c) { const t = M[p]; M[p] = M[c]; M[c] = t; }
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = M[i][n];
      for (let k = i + 1; k < n; k++) s -= M[i][k] * x[k];
      x[i] = s / M[i][i];
    }
    return x;
  }
  /** Determinante por eliminación con pivoteo parcial. */
  function det(M) {
    const n = M.length; const a = matClone(M); let d = 1;
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
      if (a[p][c] === 0) return 0;
      if (p !== c) { const t = a[p]; a[p] = a[c]; a[c] = t; d = -d; }
      d *= a[c][c];
      for (let r = c + 1; r < n; r++) {
        const f = a[r][c] / a[c][c];
        for (let k = c; k < n; k++) a[r][k] -= f * a[c][k];
      }
    }
    return d;
  }
  /** Menores principales líderes Δ₁…Δₙ (criterio de Sylvester). */
  function leadingMinors(A) {
    const out = [];
    for (let k = 1; k <= A.length; k++) out.push(det(A.slice(0, k).map((r) => r.slice(0, k))));
    return out;
  }
  /** Autovalores de una matriz simétrica por el método cíclico de Jacobi (ordenados de menor a mayor). */
  function eigenvaluesSym(S) {
    const n = S.length; const a = matClone(S);
    for (let sweep = 0; sweep < 100; sweep++) {
      let rotated = 0;
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = a[p][q];
          if (apq === 0 || Math.abs(apq) <= 1e-18 * Math.sqrt(Math.abs(a[p][p] * a[q][q]))) continue;
          const theta = (a[q][q] - a[p][p]) / (2 * apq);
          const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1); const s = t * c;
          for (let k = 0; k < n; k++) {
            const akp = a[k][p]; const akq = a[k][q];
            a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = a[p][k]; const aqk = a[q][k];
            a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
          }
          rotated++;
        }
      }
      if (rotated === 0) break;
    }
    return a.map((r, i) => r[i]).sort((x, y) => x - y);
  }

  /**
   * Gradientes Conjugados Precondicionado.
   * Si precond es null se ejecuta el GC simple (M = I).
   *   r0 = b − A x0 ;  M z0 = r0 ;  p0 = z0
   *   α_k = (r_kᵀ z_k)/(p_kᵀ A p_k) ;  x_{k+1} = x_k + α_k p_k ;  r_{k+1} = r_k − α_k A p_k
   *   M z_{k+1} = r_{k+1} ;  β_k = (r_{k+1}ᵀ z_{k+1})/(r_kᵀ z_k) ;  p_{k+1} = z_{k+1} + β_k p_k
   * Parada: ‖r_k‖₂/‖b‖₂ < tol (o ‖r_k‖₂ < tol si ‖b‖₂ = 0).
   */
  function conjugateGradient(A, b, x0, tol, maxIter, precond) {
    const n = b.length;
    const x = x0.slice();
    let r = b.map((bi, i) => bi - dot(A[i], x));
    const normb = norm2(b);
    const den = normb > 0 ? normb : 1;
    const history = [norm2(r) / den];
    const out = (extra) => Object.assign({ x, history, iterations: history.length - 1, converged: false, reason: '', breakdown: false }, extra);
    if (history[0] < tol) return out({ converged: true, reason: 'El vector inicial ya cumple la tolerancia.' });
    let z = precond ? precond(r) : r.slice();
    let p = z.slice();
    let rz = dot(r, z);
    for (let k = 1; k <= maxIter; k++) {
      if (!(rz > 0) || !Number.isFinite(rz)) return out({ breakdown: true, reason: 'Ruptura: rᵀz no es positivo (división entre cero evitada).' });
      const Ap = matVec(A, p);
      const pAp = dot(p, Ap);
      if (!(pAp > 0) || !Number.isFinite(pAp)) return out({ breakdown: true, reason: 'Ruptura: pᵀAp ≤ 0 (A no es definida positiva en la práctica o hay desbordamiento).' });
      const alpha = rz / pAp;
      for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
      const rel = norm2(r) / den;
      history.push(rel);
      if (!Number.isFinite(rel)) return out({ breakdown: true, reason: 'Ruptura: el residuo dejó de ser finito.' });
      if (rel < tol) return out({ converged: true, reason: 'Tolerancia alcanzada.' });
      z = precond ? precond(r) : r.slice();
      const rzNew = dot(r, z);
      const beta = rzNew / rz;
      for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
      rz = rzNew;
    }
    return out({ reason: 'Se alcanzó el máximo de iteraciones sin cumplir la tolerancia.' });
  }

  function errorTable(xref, x) {
    return xref.map((xr, i) => {
      const ea = Math.abs(xr - x[i]);
      const defined = Math.abs(xr) > 1e-12;
      const er = defined ? ea / Math.abs(xr) : null;
      return { ea, er, ep: er === null ? null : er * 100 };
    });
  }

  /* ===================================================================
     3. VALIDACIÓN Y RESOLUCIÓN
     =================================================================== */
  function validateSystem(sys) {
    const errors = []; const warnings = [];
    const bad = { A: new Set(), b: new Set(), x0: new Set(), tol: false, maxIter: false };
    const info = {};
    const n = sys.n;
    if (!Number.isInteger(n) || n < 4) errors.push('El sistema debe tener n ≥ 4 ecuaciones e incógnitas.');

    const badA = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (!Number.isFinite(sys.A[i][j])) { bad.A.add(i + ',' + j); badA.push('a' + sub(i) + sub(j)); }
    }
    if (badA.length) errors.push('La matriz A tiene ' + badA.length + ' entrada(s) vacía(s) o no numérica(s): ' + badA.slice(0, 4).join(', ') + (badA.length > 4 ? '…' : '') + '.');
    const badB = [];
    for (let i = 0; i < n; i++) if (!Number.isFinite(sys.b[i])) { bad.b.add(i); badB.push('b' + sub(i)); }
    if (badB.length) errors.push('El vector b tiene entrada(s) vacía(s) o no numérica(s): ' + badB.join(', ') + '.');
    const badX = [];
    for (let i = 0; i < n; i++) if (!Number.isFinite(sys.x0[i])) { bad.x0.add(i); badX.push('x₀' + sub(i)); }
    if (badX.length) errors.push('El vector inicial x₀ tiene entrada(s) no válida(s): ' + badX.join(', ') + '.');
    if (!(Number.isFinite(sys.tol) && sys.tol > 0 && sys.tol < 1)) { bad.tol = true; errors.push('La tolerancia debe cumplir 0 < tol < 1 (por ejemplo 1e-6).'); }
    if (!(Number.isInteger(sys.maxIter) && sys.maxIter >= 1 && sys.maxIter <= 100000)) { bad.maxIter = true; errors.push('El máximo de iteraciones debe ser un entero entre 1 y 100000.'); }

    if (badA.length === 0 && n >= 2) {
      // Simetría
      const asym = []; let maxAsym = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const d = Math.abs(sys.A[i][j] - sys.A[j][i]);
        const sc = Math.max(Math.abs(sys.A[i][j]), Math.abs(sys.A[j][i]));
        maxAsym = Math.max(maxAsym, d);
        if (d > 1e-12 * sc) { asym.push([i, j]); bad.A.add(i + ',' + j); bad.A.add(j + ',' + i); }
      }
      info.symmetric = asym.length === 0; info.maxAsym = maxAsym;
      if (!info.symmetric) {
        const [i, j] = asym[0];
        errors.push('A no es simétrica: a' + sub(i) + sub(j) + ' = ' + num(sys.A[i][j], 6) + ' ≠ a' + sub(j) + sub(i) + ' = ' + num(sys.A[j][i], 6) + (asym.length > 1 ? ' (y ' + (asym.length - 1) + ' par(es) más)' : '') + '. GC/GCP exigen A = Aᵀ.');
      } else {
        const minors = leadingMinors(sys.A);
        const chol = cholesky(sys.A);
        const eig = eigenvaluesSym(sys.A);
        info.minors = minors; info.eigenvalues = eig; info.cholOk = chol.ok;
        info.pivots = chol.ok ? chol.L.map((r, i) => r[i] * r[i]) : null;
        const pd = chol.ok && minors.every((m) => m > 0) && eig[0] > 0;
        info.positiveDefinite = pd;
        if (!pd) {
          const why = [];
          if (eig[0] <= 0) why.push('λmin = ' + num(eig[0], 6) + ' ≤ 0');
          const k = minors.findIndex((m) => !(m > 0));
          if (k >= 0) why.push('menor principal Δ' + sub(k) + ' = ' + num(minors[k], 6) + ' ≤ 0');
          if (!chol.ok) why.push('Cholesky falla en la fila ' + (chol.index + 1) + ' (pivote no positivo)');
          errors.push('A es simétrica pero NO es definida positiva: ' + why.join('; ') + '.');
        } else {
          info.L = chol.L;
          info.lambdaMin = eig[0]; info.lambdaMax = eig[n - 1];
          info.kappaA = eig[n - 1] / eig[0];
          const B = sys.A.map((r, i) => r.map((v, j) => v / Math.sqrt(sys.A[i][i] * sys.A[j][j])));
          const eigB = eigenvaluesSym(B);
          info.eigenvaluesPre = eigB;
          info.kappaM = eigB[n - 1] / eigB[0];
        }
      }
    }
    if (errors.length === 0 && sys.b.every((v) => v === 0)) warnings.push('b = 0: la solución es x = 0 y el residuo relativo no está definido; se usa el residuo absoluto.');
    return { ok: errors.length === 0, errors, warnings, bad, info };
  }

  function relResidual(A, b, x) {
    const nb = norm2(b);
    return norm2(b.map((bi, i) => bi - dot(A[i], x))) / (nb > 0 ? nb : 1);
  }

  function solveSystem(sys, validation) {
    const val = validation || validateSystem(sys);
    if (!val.ok) return { ok: false, errors: val.errors, validation: val };
    const { A, b, x0, tol, maxIter } = sys;
    const xref = choleskySolve(val.info.L, b);
    const d = A.map((r, i) => r[i]);
    const jacobi = (r) => r.map((ri, i) => ri / d[i]);
    const wrap = (m) => Object.assign(m, {
      trueRes: relResidual(A, b, m.x),
      errors: errorTable(xref, m.x),
      relErr: norm2(m.x.map((v, i) => v - xref[i])) / (norm2(xref) || 1)
    });
    const gc = wrap(conjugateGradient(A, b, x0, tol, maxIter, null));
    const gcp = wrap(conjugateGradient(A, b, x0, tol, maxIter, jacobi));
    const warnings = val.warnings.slice();
    if (xref.some((v) => v < 0)) warnings.push('Regla de negocio: la solución de referencia tiene asignaciones negativas; revise b y A (un presupuesto negativo no tiene sentido financiero).');
    return {
      ok: true, sys, validation: val, tol, maxIter, warnings, xref,
      refRes: relResidual(A, b, xref), gc, gcp,
      kappaA: val.info.kappaA, kappaM: val.info.kappaM,
      lambdaMin: val.info.lambdaMin, lambdaMax: val.info.lambdaMax,
      lambdaMinPre: val.info.eigenvaluesPre[0], lambdaMaxPre: val.info.eigenvaluesPre[sys.n - 1]
    };
  }

  /* ===================================================================
     4. CASO FINANCIERO Y ESCENARIOS
     =================================================================== */
  const VARS = [
    { corto: 'Infra', nombre: 'Infraestructura', icono: '🖥️', desc: 'Servidores, redes, centro de datos y nube. Alta intensidad de capital (CAPEX) y contratos indexados a divisas.' },
    { corto: 'Desarrollo', nombre: 'Desarrollo de software', icono: '💻', desc: 'Equipos de ingeniería, licencias y herramientas. Principal componente de gasto en personal.' },
    { corto: 'Seguridad', nombre: 'Seguridad informática', icono: '🛡️', desc: 'Prevención y respuesta ante incidentes, auditorías, cumplimiento normativo y monitoreo.' },
    { corto: 'Soporte', nombre: 'Soporte técnico', icono: '🎧', desc: 'Mesa de ayuda, mantenimiento y atención a usuarios internos. Gasto principalmente operativo (OPEX).' }
  ];

  const SCENARIOS = {
    ideal: {
      key: 'ideal', nombre: 'Ideal', tono: 'ok',
      descripcion: 'Situación estable y equilibrada: costos y demandas proporcionales, interdependencias moderadas entre áreas.',
      A: [[10, 2, 1, 1], [2, 12, 2, 1], [1, 2, 9, 2], [1, 1, 2, 8]],
      b: [9000, 14400, 6300, 4800],
      justificacion: [
        'Diagonal (10, 12, 9, 8): costo marginal ponderado de cada departamento; Desarrollo es el mayor por su intensidad en personal y Soporte el menor.',
        'Fuera de la diagonal (1 a 2): interdependencia. Infraestructura–Desarrollo y Desarrollo–Seguridad y Seguridad–Soporte valen 2 (entornos de prueba y nube, ciclo de desarrollo seguro, respuesta a incidentes); el resto vale 1.',
        'b = demanda ponderada de presupuesto: Desarrollo 14 400, Infraestructura 9 000, Seguridad 6 300, Soporte 4 800.',
        'La matriz es estrictamente diagonal dominante: es simétrica definida positiva y tiene κ(A) bajo.'
      ]
    },
    estres: {
      key: 'estres', nombre: 'Bajo estrés financiero', tono: 'warn',
      descripcion: 'Alza de costos importados por tipo de cambio e inflación, recorte de ingresos y un incidente de ciberseguridad.',
      A: [[13, 2, 1.5, 1], [2, 13.2, 2, 1], [1.5, 2, 10.8, 2.5], [1, 1, 2.5, 8.8]],
      b: [10350, 13680, 8190, 5520],
      justificacion: [
        'a₁₁: 10 → 13 (+30 %): equipos y contratos de nube importados se encarecen por el tipo de cambio.',
        'a₂₂: 12 → 13.2 (+10 %): licencias y herramientas cotizadas en dólares. a₃₃: 9 → 10.8 (+20 %): servicios de seguridad especializados. a₄₄: 8 → 8.8 (+10 %): salarios por inflación.',
        'Acoplamientos: a₁₃ = a₃₁ = 1 → 1.5 (endurecimiento de la infraestructura) y a₃₄ = a₄₃ = 2 → 2.5 (más coordinación en la respuesta a incidentes).',
        'b: Infraestructura +15 % (inflación), Desarrollo −5 % (menor ingreso), Seguridad +30 % (incidente), Soporte +15 % (más tickets).',
        'A sigue siendo diagonal dominante, por lo que se mantiene simétrica definida positiva.'
      ]
    },
    mal: {
      key: 'mal', nombre: 'Mal condicionado', tono: 'bad',
      descripcion: 'Las ponderaciones de costo de cada área difieren en órdenes de magnitud (mal escalamiento diagonal) con acoplamientos moderados.',
      A: [[10000, 100, 400, 5], [100, 16, 20, 0.3], [400, 20, 400, 3], [5, 0.3, 3, 0.25]],
      b: [9422500, 126550, 669500, 7145],
      justificacion: [
        'Diagonal (10 000, 16, 400, 0.25): la infraestructura (CAPEX, contratos en divisas) tiene una curvatura de costo miles de veces mayor que la del soporte (OPEX pequeño y flexible).',
        'Construcción: A = S·C·S con S = diag(100, 4, 20, 0.5) y C una matriz de correlaciones diagonal dominante (0.1 a 0.3). Es congruente con una SPD, luego es SPD.',
        'b = A·x* con x* = (900, 1400, 700, 500): magnitudes de 7 145 a 9 422 500 que reflejan las distintas escalas.',
        'Como el mal condicionamiento es de escala diagonal, corresponde el precondicionador de Jacobi (M = diag(A)).'
      ]
    }
  };
  const SCEN_KEYS = ['ideal', 'estres', 'mal'];

  function scenarioSystem(key, tol, maxIter) {
    const s = SCENARIOS[key];
    return { n: 4, A: matClone(s.A), b: s.b.slice(), x0: [0, 0, 0, 0], tol, maxIter };
  }

  /* ===================================================================
     5. ESTADO Y EDITOR
     =================================================================== */
  const state = { scenarioKey: 'ideal', modified: false, lastResult: null, all: null, chart: null };
  let tolInput, maxIterInput, symChk;

  function buildEditor() {
    const n = VARS.length;
    let h = '<table class="matrix-table"><thead><tr><th scope="col" class="corner">Ecuación</th>';
    for (let j = 0; j < n; j++) h += '<th scope="col">x' + sub(j) + '<small>' + esc(VARS[j].corto) + '</small></th>';
    h += '<th scope="col" class="sep">b<small>miles de Bs</small></th></tr></thead><tbody>';
    for (let i = 0; i < n; i++) {
      h += '<tr><th scope="row">' + (i + 1) + '. ' + esc(VARS[i].corto) + '</th>';
      for (let j = 0; j < n; j++) h += '<td><input class="cell" id="a-' + i + '-' + j + '" data-i="' + i + '" data-j="' + j + '" type="text" inputmode="decimal" autocomplete="off" aria-label="Coeficiente a' + (i + 1) + (j + 1) + '"></td>';
      h += '<td class="sep"><input class="cell" id="b-' + i + '" type="text" inputmode="decimal" autocomplete="off" aria-label="Término independiente b' + (i + 1) + '"></td></tr>';
    }
    h += '</tbody></table>';
    $('editor-matriz').innerHTML = h;
    let x = '<table class="matrix-table x0-table"><thead><tr>';
    for (let j = 0; j < n; j++) x += '<th scope="col">x₀' + sub(j) + '<small>' + esc(VARS[j].corto) + '</small></th>';
    x += '</tr></thead><tbody><tr>';
    for (let j = 0; j < n; j++) x += '<td><input class="cell" id="x0-' + j + '" type="text" inputmode="decimal" autocomplete="off" aria-label="Vector inicial x0 componente ' + (j + 1) + '"></td>';
    x += '</tr></tbody></table>';
    $('editor-x0').innerHTML = x;
  }

  function setEditor(A, b, x0) {
    for (let i = 0; i < VARS.length; i++) {
      for (let j = 0; j < VARS.length; j++) $('a-' + i + '-' + j).value = String(A[i][j]);
      $('b-' + i).value = String(b[i]);
      $('x0-' + i).value = String(x0[i]);
    }
  }
  function readSystem() {
    const n = VARS.length; const A = []; const b = []; const x0 = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) row.push(parseNumber($('a-' + i + '-' + j).value));
      A.push(row); b.push(parseNumber($('b-' + i).value)); x0.push(parseNumber($('x0-' + i).value));
    }
    return { n, A, b, x0, tol: parseNumber(tolInput.value), maxIter: parseNumber(maxIterInput.value) };
  }
  function snapshotEditor() {
    const vals = {};
    document.querySelectorAll('#editor input.cell').forEach((el) => { vals[el.id] = el.value; });
    return { vals, tol: tolInput.value, maxIter: maxIterInput.value, sym: symChk.checked, key: state.scenarioKey, modified: state.modified };
  }
  function restoreEditor(s) {
    Object.keys(s.vals).forEach((id) => { $(id).value = s.vals[id]; });
    tolInput.value = s.tol; maxIterInput.value = s.maxIter; symChk.checked = s.sym;
    recalc(); state.scenarioKey = s.key; state.modified = s.modified; updateScenarioUI(); renderComparativa();
  }
  function setInput(id, value) {
    const el = $(id); el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function loadScenario(key) {
    const s = SCENARIOS[key];
    setEditor(s.A, s.b, [0, 0, 0, 0]);
    state.scenarioKey = key; state.modified = false;
    recalc(); updateScenarioUI();
    return state.lastResult;
  }
  function updateScenarioUI() {
    const s = SCENARIOS[state.scenarioKey];
    const label = s.nombre + (state.modified ? ' (modificado manualmente)' : '');
    ['badge-escenario', 'badge-escenario-2'].forEach((id) => { const el = $(id); if (el) { el.textContent = 'Escenario activo: ' + label; el.className = 'badge ' + (state.modified ? 'neutral' : s.tono); } });
    document.querySelectorAll('.scen-card').forEach((c) => c.classList.toggle('active', c.dataset.key === state.scenarioKey && !state.modified));
  }

  function onEditorInput(e) {
    const t = e.target;
    if (!t.classList || !t.classList.contains('cell') && t !== tolInput && t !== maxIterInput) return;
    if (t.dataset && t.dataset.i !== undefined && symChk.checked && t.dataset.i !== t.dataset.j) {
      const mirror = $('a-' + t.dataset.j + '-' + t.dataset.i);
      if (mirror) mirror.value = t.value;
    }
    if (t === tolInput || t === maxIterInput) { renderComparativa(); } else { state.modified = true; updateScenarioUI(); }
    recalc();
  }

  /* ===================================================================
     6. RECÁLCULO Y RENDER
     =================================================================== */
  function setResultsVisible(v) {
    document.querySelectorAll('.js-result').forEach((el) => { el.hidden = !v; });
    document.querySelectorAll('.js-placeholder').forEach((el) => { el.hidden = v; });
  }

  function recalc() {
    const sys = readSystem();
    const val = validateSystem(sys);
    document.querySelectorAll('#editor input.cell').forEach((el) => el.classList.remove('invalid'));
    tolInput.classList.toggle('invalid', val.bad.tol); maxIterInput.classList.toggle('invalid', val.bad.maxIter);
    val.bad.A.forEach((k) => { const [i, j] = k.split(','); $('a-' + i + '-' + j).classList.add('invalid'); });
    val.bad.b.forEach((i) => $('b-' + i).classList.add('invalid'));
    val.bad.x0.forEach((i) => $('x0-' + i).classList.add('invalid'));
    renderFormulacion(sys, val);
    const box = $('editor-errores');
    if (!val.ok) {
      state.lastResult = null;
      box.hidden = false;
      box.className = 'alert error';
      box.innerHTML = '<strong>No se puede ejecutar: corrija los datos.</strong><ul>' + val.errors.map((e) => '<li>' + esc(e) + '</li>').join('') + '</ul>';
      setResultsVisible(false); drawChart();
      return null;
    }
    const res = solveSystem(sys, val);
    state.lastResult = res;
    if (res.warnings.length || !res.gc.converged || !res.gcp.converged) {
      const w = res.warnings.slice();
      if (!res.gc.converged) w.push('GC simple no convergió: ' + res.gc.reason);
      if (!res.gcp.converged) w.push('GCP no convergió: ' + res.gcp.reason);
      box.hidden = false; box.className = 'alert warn';
      box.innerHTML = '<strong>Avisos</strong><ul>' + w.map((e) => '<li>' + esc(e) + '</li>').join('') + '</ul>';
    } else {
      box.hidden = false; box.className = 'alert ok';
      box.innerHTML = '<strong>Datos válidos.</strong> A es simétrica y definida positiva; se recalcularon la solución de referencia, GC y GCP.';
    }
    setResultsVisible(true);
    renderResultados(res);
    drawChart();
    return res;
  }

  const cellTxt = (v) => (Number.isFinite(v) ? num(v, 4).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '') : '?');
  function eqText(sys, i) {
    let s = '';
    for (let j = 0; j < sys.n; j++) {
      const a = sys.A[i][j];
      if (!Number.isFinite(a)) { s += (j ? ' + ' : '') + '?·x' + sub(j); continue; }
      const mag = cellTxt(Math.abs(a));
      if (j === 0) s += (a < 0 ? '−' : '') + mag + '·x' + sub(j);
      else s += (a < 0 ? ' − ' : ' + ') + mag + '·x' + sub(j);
    }
    return s + ' = ' + cellTxt(sys.b[i]);
  }

  function renderFormulacion(sys, val) {
    const n = sys.n;
    let m = '<div class="mat-eq"><div class="mat" style="--cols:' + n + '">';
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) m += '<span class="' + (val.bad.A.has(i + ',' + j) ? 'bad' : '') + '">' + cellTxt(sys.A[i][j]) + '</span>';
    m += '</div><span class="op">·</span><div class="mat vec" style="--cols:1">';
    for (let i = 0; i < n; i++) m += '<span>x' + sub(i) + '</span>';
    m += '</div><span class="op">=</span><div class="mat vec" style="--cols:1">';
    for (let i = 0; i < n; i++) m += '<span class="' + (val.bad.b.has(i) ? 'bad' : '') + '">' + cellTxt(sys.b[i]) + '</span>';
    m += '</div></div>';
    $('form-matriz').innerHTML = m;
    $('form-sistema').innerHTML = '<ol class="eqs">' + sys.A.map((_, i) => '<li><span>' + esc(eqText(sys, i)) + '</span><small>' + esc(VARS[i].nombre) + '</small></li>').join('') + '</ol>';

    const i = val.info; let h = '';
    const row = (ok, title, detail) => '<li class="' + (ok ? 'pass' : 'fail') + '"><span class="mark">' + (ok ? '✔' : '✘') + '</span><div><strong>' + title + '</strong><div>' + detail + '</div></div></li>';
    if (i.symmetric === undefined) {
      h = '<p class="muted">La verificación de simetría y definición positiva se realizará cuando todas las entradas de A sean números válidos.</p>';
    } else {
      h += '<ul class="checks">';
      h += row(i.symmetric, 'Simetría A = Aᵀ', 'máx |aᵢⱼ − aⱼᵢ| = ' + num(i.maxAsym, 6));
      if (i.minors) {
        h += row(i.minors.every((x) => x > 0), 'Criterio de Sylvester (menores principales Δₖ > 0)', i.minors.map((x, k) => 'Δ' + sub(k) + ' = ' + num(x, 4)).join(' · '));
        h += row(i.eigenvalues[0] > 0, 'Autovalores positivos (método de Jacobi)', i.eigenvalues.map((x, k) => 'λ' + sub(k) + ' = ' + num(x, 5)).join(' · '));
        h += row(i.cholOk, 'Factorización de Cholesky A = LLᵀ', i.cholOk ? 'todos los pivotes son positivos: ' + i.pivots.map((x) => num(x, 4)).join(', ') : 'falló: algún pivote no es positivo');
      }
      h += '</ul>';
      h += val.ok ? '<p class="verdict pass">A es simétrica definida positiva: el sistema puede resolverse con GC/GCP. κ(A) = ' + num(i.kappaA, 4) + '.</p>'
        : '<p class="verdict fail">A no cumple las hipótesis del método: la ejecución está bloqueada.</p>';
    }
    $('form-spd').innerHTML = h;
  }

  function vecRows(res) {
    return res.xref.map((x, i) => '<tr><th scope="row">x' + sub(i) + ' · ' + esc(VARS[i].nombre) + '</th><td data-v="ref-' + i + '">' + num(x, 6) + '</td><td data-v="gc-' + i + '">' + num(res.gc.x[i], 6) + '</td><td data-v="gcp-' + i + '">' + num(res.gcp.x[i], 6) + '</td></tr>').join('');
  }
  function errRows(res, m) {
    return res.xref.map((x, i) => {
      const e = m.errors[i];
      return '<tr><th scope="row">x' + sub(i) + ' · ' + esc(VARS[i].corto) + '</th><td>' + num(x, 6) + '</td><td>' + num(m.x[i], 6) + '</td><td>' + sci(e.ea) + '</td><td>' + (e.er === null ? 'no definido' : sci(e.er)) + '</td><td>' + (e.ep === null ? 'no definido' : sci(e.ep) + ' %') + '</td></tr>';
    }).join('');
  }
  const yn = (v) => (v ? '<span class="tag ok">Sí</span>' : '<span class="tag bad">No</span>');

  function renderResultados(res) {
    const total = res.xref.reduce((a, v) => a + v, 0);
    // --- Sección 6
    let k = '<div class="kpis">';
    const kpi = (t, v, s, c) => '<div class="kpi ' + (c || '') + '"><span>' + t + '</span><strong>' + v + '</strong><small>' + s + '</small></div>';
    k += kpi('Solución de referencia', 'Cholesky', 'método directo · residuo ' + sci(res.refRes), 'ref');
    k += kpi('GC simple', res.gc.iterations + ' it.', (res.gc.converged ? 'convergió' : 'NO convergió') + ' · κ(A) = ' + num(res.kappaA, 3), 'gc');
    k += kpi('GCP · Jacobi', res.gcp.iterations + ' it.', (res.gcp.converged ? 'convergió' : 'NO convergió') + ' · κ(M⁻¹A) = ' + num(res.kappaM, 3), 'gcp');
    k += '</div>';
    $('kpis').innerHTML = k;
    $('tabla-comparacion').innerHTML = '<thead><tr><th>Variable (miles de Bs)</th><th>Referencia (Cholesky)</th><th>GC</th><th>GCP</th></tr></thead><tbody>' + vecRows(res) + '</tbody><tfoot><tr><th>Total Σx</th><td>' + num(total, 6) + '</td><td>' + num(res.gc.x.reduce((a, v) => a + v, 0), 6) + '</td><td>' + num(res.gcp.x.reduce((a, v) => a + v, 0), 6) + '</td></tr></tfoot>';
    const maxv = Math.max.apply(null, res.xref.map((v) => Math.abs(v))) || 1;
    $('barras').innerHTML = res.xref.map((x, i) => '<div class="bar-row"><span class="bar-name">' + esc(VARS[i].nombre) + '</span><div class="bar-track"><div class="bar-fill c' + i + '" style="width:' + Math.max(0, Math.min(100, (Math.abs(x) / maxv) * 100)).toFixed(1) + '%"></div></div><span class="bar-val">' + num(x, 2) + ' · ' + (total ? num((x / total) * 100, 1) : '0') + ' %</span></div>').join('');
    const diff = res.gc.iterations - res.gcp.iterations;
    let txt = 'Con los datos actuales, la solución de referencia (método directo) asigna un total de <strong>' + num(total, 2) + '</strong> miles de Bs. ';
    txt += 'GC necesitó <strong>' + res.gc.iterations + '</strong> iteraciones y GCP con Jacobi <strong>' + res.gcp.iterations + '</strong>';
    txt += diff > 0 ? ' (GCP ahorra ' + diff + ').' : diff === 0 ? ' (el mismo número).' : ' (GCP necesitó más).';
    txt += ' El precondicionador pasó el número de condición de ' + num(res.kappaA, 3) + ' a ' + num(res.kappaM, 3) + '.';
    $('explicacion').innerHTML = txt;

    // --- Sección 7
    $('tabla-err-gcp').innerHTML = '<thead><tr><th>Variable</th><th>x_ref</th><th>x_GCP</th><th>Error abs. Eₐ</th><th>Error rel. Eᵣ</th><th>Error % Eₚ</th></tr></thead><tbody>' + errRows(res, res.gcp) + '</tbody>';
    $('tabla-err-gc').innerHTML = '<thead><tr><th>Variable</th><th>x_ref</th><th>x_GC</th><th>Error abs. Eₐ</th><th>Error rel. Eᵣ</th><th>Error % Eₚ</th></tr></thead><tbody>' + errRows(res, res.gc) + '</tbody>';
    const mrow = (name, m, extra) => '<tr><th scope="row">' + name + '</th><td>' + extra[0] + '</td><td>' + extra[1] + '</td><td>' + extra[2] + '</td><td>' + extra[3] + '</td><td>' + extra[4] + '</td></tr>';
    $('tabla-metodos').innerHTML = '<thead><tr><th>Método</th><th>Iteraciones</th><th>¿Convergió?</th><th>Residuo relativo final ‖b−Ax‖/‖b‖</th><th>‖x−x_ref‖/‖x_ref‖</th><th>Motivo de parada</th></tr></thead><tbody>'
      + mrow('Cholesky (referencia)', null, ['—', yn(true), sci(res.refRes), '0', 'Método directo'])
      + mrow('GC simple', res.gc, [res.gc.iterations, yn(res.gc.converged), sci(res.gc.trueRes), sci(res.gc.relErr), esc(res.gc.reason)])
      + mrow('GCP (Jacobi)', res.gcp, [res.gcp.iterations, yn(res.gcp.converged), sci(res.gcp.trueRes), sci(res.gcp.relErr), esc(res.gcp.reason)]) + '</tbody>';
    const rho = (kp) => (Math.sqrt(kp) - 1) / (Math.sqrt(kp) + 1);
    $('tabla-cond').innerHTML = '<thead><tr><th>Matriz</th><th>λ_min</th><th>λ_max</th><th>Número de condición κ</th><th>Factor (√κ−1)/(√κ+1)</th></tr></thead><tbody>'
      + '<tr><th scope="row">A (sin precondicionar)</th><td>' + num(res.lambdaMin, 6) + '</td><td>' + num(res.lambdaMax, 6) + '</td><td data-v="kappaA">' + num(res.kappaA, 4) + '</td><td>' + num(rho(res.kappaA), 6) + '</td></tr>'
      + '<tr><th scope="row">M⁻¹A (Jacobi, M = diag A)</th><td>' + num(res.lambdaMinPre, 6) + '</td><td>' + num(res.lambdaMaxPre, 6) + '</td><td data-v="kappaM">' + num(res.kappaM, 4) + '</td><td>' + num(rho(res.kappaM), 6) + '</td></tr></tbody>';
    $('nota-cond').textContent = 'κ(A) = λmax/λmin (norma 2, A simétrica definida positiva). M⁻¹A no es simétrica pero es semejante a D^(−1/2)·A·D^(−1/2), de modo que tiene los mismos autovalores y κ(M⁻¹A) se calcula con ellos.';
    // tabla de iteraciones
    const L = Math.max(res.gc.history.length, res.gcp.history.length); const cap = Math.min(L, 201);
    let it = '<thead><tr><th>k</th><th>GC: ‖rₖ‖/‖b‖</th><th>GCP: ‖rₖ‖/‖b‖</th></tr></thead><tbody>';
    for (let q = 0; q < cap; q++) it += '<tr><th scope="row">' + q + '</th><td>' + (q < res.gc.history.length ? sci(res.gc.history[q], 4) : '—') + '</td><td>' + (q < res.gcp.history.length ? sci(res.gcp.history[q], 4) : '—') + '</td></tr>';
    $('tabla-iter').innerHTML = it + '</tbody>';
    $('nota-iter').textContent = L > cap ? 'Se muestran las primeras ' + (cap - 1) + ' iteraciones de ' + (L - 1) + '.' : 'Residuo relativo recursivo del algoritmo en cada iteración.';
    // resumen para el gráfico
    $('resumen-grafico').textContent = 'GC: ' + res.gc.iterations + ' iteraciones (residuo final ' + sci(res.gc.history[res.gc.history.length - 1]) + '). GCP: ' + res.gcp.iterations + ' iteraciones (residuo final ' + sci(res.gcp.history[res.gcp.history.length - 1]) + '). Tolerancia: ' + sci(res.tol, 1) + '.';
  }

  /* ---------- Tabla comparativa de los 3 escenarios y conclusiones ---------- */
  function computeAll(tol, maxIter) {
    const all = {};
    SCEN_KEYS.forEach((k) => { const sys = scenarioSystem(k, tol, maxIter); all[k] = solveSystem(sys, validateSystem(sys)); });
    return all;
  }
  function vecList(arr, f) { return '<ul class="vl">' + arr.map((v, i) => '<li>x' + sub(i) + ' = ' + f(v, i) + '</li>').join('') + '</ul>'; }

  function interpretacion(key, all) {
    const r = all[key]; const total = r.xref.reduce((a, v) => a + v, 0);
    if (key === 'ideal') {
      const im = r.xref.indexOf(Math.max.apply(null, r.xref));
      return 'Presupuesto total óptimo Σx = ' + num(total, 2) + ' miles de Bs; la mayor asignación es ' + VARS[im].nombre + ' (' + num((r.xref[im] / total) * 100, 1) + ' %). Con κ(A) = ' + num(r.kappaA, 3) + ' el sistema es estable: variaciones pequeñas de b producen variaciones proporcionales de x.';
    }
    if (key === 'estres') {
      const base = all.ideal; const bt = base.xref.reduce((a, v) => a + v, 0);
      const ch = r.xref.map((v, i) => (v - base.xref[i]) / base.xref[i] * 100);
      const im = ch.reduce((m, v, i) => (Math.abs(v) > Math.abs(ch[m]) ? i : m), 0);
      const ratio = r.kappaA / base.kappaA;
      const cond = ratio >= 0.5 && ratio <= 2 ? 'el condicionamiento casi no cambia (κ de ' + num(base.kappaA, 3) + ' a ' + num(r.kappaA, 3) + '): el estrés desplaza el punto de equilibrio sin deteriorar la estabilidad numérica.' : 'el condicionamiento cambia de forma apreciable (κ de ' + num(base.kappaA, 3) + ' a ' + num(r.kappaA, 3) + ').';
      return 'Frente al escenario ideal el presupuesto total óptimo pasa de ' + num(bt, 2) + ' a ' + num(total, 2) + ' (' + (total >= bt ? '+' : '−') + num(Math.abs((total - bt) / bt * 100), 1) + ' %). La mayor variación relativa es ' + VARS[im].nombre + ' (' + (ch[im] >= 0 ? '+' : '−') + num(Math.abs(ch[im]), 1) + ' %). Además, ' + cond;
    }
    const diff = r.gc.iterations - r.gcp.iterations;
    const g = diff > 0 ? 'GCP ahorra ' + diff + ' iteración(es) respecto de GC (' + r.gcp.iterations + ' frente a ' + r.gc.iterations + ').' : diff === 0 ? 'GC y GCP necesitan el mismo número de iteraciones (' + r.gc.iterations + '); con n = 4 el método termina en pocos pasos por su propiedad de terminación finita, así que la ganancia se ve sobre todo en el condicionamiento.' : 'GCP necesitó más iteraciones (' + r.gcp.iterations + ') que GC (' + r.gc.iterations + ').';
    return 'La escala de las ponderaciones eleva κ(A) a ' + num(r.kappaA, 3) + '; con Jacobi κ(M⁻¹A) = ' + num(r.kappaM, 3) + ' (reducción de ×' + num(r.kappaA / r.kappaM, 3) + '). ' + g;
  }

  function renderComparativa() {
    const host = $('tabla-comparativa-host');
    const tol = parseNumber(tolInput.value); const mi = parseNumber(maxIterInput.value);
    if (!(tol > 0 && tol < 1) || !Number.isInteger(mi) || mi < 1 || mi > 100000) {
      host.innerHTML = '<div class="alert error">La tolerancia o el máximo de iteraciones no son válidos; corríjalos en el editor para recalcular la tabla comparativa.</div>';
      state.all = null; $('conclusiones-lista').innerHTML = ''; return;
    }
    const all = computeAll(tol, mi); state.all = all;
    if (SCEN_KEYS.some((k) => !all[k].ok)) { host.innerHTML = '<div class="alert error">Algún escenario no es válido.</div>'; return; }
    const col = (f) => SCEN_KEYS.map((k) => '<td>' + f(all[k], k) + '</td>').join('');
    let t = '<table class="cmp"><thead><tr><th>Indicador</th>' + SCEN_KEYS.map((k) => '<th>' + SCENARIOS[k].nombre + '</th>').join('') + '</tr></thead><tbody>';
    t += '<tr><th scope="row">Solución numérica de referencia (Cholesky)</th>' + col((r) => vecList(r.xref, (v) => num(v, 6))) + '</tr>';
    t += '<tr><th scope="row">Solución GCP</th>' + col((r) => vecList(r.gcp.x, (v) => num(v, 6))) + '</tr>';
    t += '<tr><th scope="row">Error absoluto Eₐ (GCP)</th>' + col((r) => vecList(r.gcp.errors, (e) => sci(e.ea))) + '</tr>';
    t += '<tr><th scope="row">Error porcentual Eₚ (GCP)</th>' + col((r) => vecList(r.gcp.errors, (e) => (e.ep === null ? 'no definido' : sci(e.ep) + ' %'))) + '</tr>';
    t += '<tr><th scope="row">Iteraciones GC (sin precondicionar)</th>' + col((r) => '<strong>' + r.gc.iterations + '</strong>') + '</tr>';
    t += '<tr><th scope="row">Iteraciones GCP (precondicionado)</th>' + col((r) => '<strong>' + r.gcp.iterations + '</strong>') + '</tr>';
    t += '<tr><th scope="row">Número de condición κ(A)</th>' + col((r) => num(r.kappaA, 4)) + '</tr>';
    t += '<tr><th scope="row">Número de condición κ(M⁻¹A)</th>' + col((r) => num(r.kappaM, 4)) + '</tr>';
    t += '<tr><th scope="row">Residuo relativo final GCP</th>' + col((r) => sci(r.gcp.trueRes)) + '</tr>';
    t += '<tr><th scope="row">¿Convergió el algoritmo? (GC / GCP)</th>' + col((r) => yn(r.gc.converged) + ' / ' + yn(r.gcp.converged)) + '</tr>';
    t += '<tr><th scope="row">Interpretación financiera</th>' + col((r, k) => '<span class="interp">' + esc(interpretacion(k, all)) + '</span>') + '</tr>';
    t += '</tbody></table>';
    host.innerHTML = '<div class="table-wrap">' + t + '</div>';
    renderConclusiones(all);
  }

  function renderConclusiones(all) {
    const items = [];
    const nm = (k) => SCENARIOS[k].nombre;
    items.push('<strong>Validez del modelo.</strong> Los tres escenarios son simétricos y definidos positivos (Sylvester, autovalores y Cholesky). Los κ(A) obtenidos son ' + SCEN_KEYS.map((k) => nm(k) + ': ' + num(all[k].kappaA, 3)).join('; ') + '.');
    items.push('<strong>Exactitud.</strong> El mayor error porcentual de GCP respecto de la referencia directa fue ' + SCEN_KEYS.map((k) => { const m = Math.max.apply(null, all[k].gcp.errors.map((e) => (e.ep === null ? 0 : e.ep))); return nm(k) + ': ' + sci(m) + ' %'; }).join('; ') + '. Con la tolerancia ' + sci(all.ideal.tol, 1) + ' sobre el residuo relativo, el error de la solución queda acotado por κ(A)·tol.');
    items.push('<strong>Convergencia.</strong> Iteraciones GC / GCP: ' + SCEN_KEYS.map((k) => nm(k) + ' ' + all[k].gc.iterations + ' / ' + all[k].gcp.iterations).join('; ') + '. ' + (SCEN_KEYS.every((k) => all[k].gc.converged && all[k].gcp.converged) ? 'Ambos métodos convergieron en los tres escenarios.' : 'Al menos un método no convergió; revise el máximo de iteraciones.'));
    const m = all.mal; const d = m.gc.iterations - m.gcp.iterations;
    items.push('<strong>Efecto del precondicionador.</strong> En el escenario mal condicionado Jacobi reduce el número de condición de ' + num(m.kappaA, 3) + ' a ' + num(m.kappaM, 3) + ' (×' + num(m.kappaA / m.kappaM, 3) + '). ' + (d > 0 ? 'GCP ahorra ' + d + ' iteración(es) frente a GC.' : d === 0 ? 'En iteraciones no hay diferencia: con n = 4, GC termina en pocas iteraciones aunque κ sea alto, de modo que la ventaja del precondicionador se aprecia en el condicionamiento y crecería con sistemas mayores.' : 'En este caso GCP no ahorró iteraciones.'));
    const e = all.estres, id = all.ideal;
    items.push('<strong>Impacto financiero del estrés.</strong> ' + interpretacion('estres', all));
    items.push('<strong>Limitaciones.</strong> Los datos son sintéticos y didácticos (no provienen de una empresa real); el modelo es cuadrático (costo ½xᵀAx − bᵀx); el precondicionador de Jacobi corrige el mal escalamiento diagonal, no una fuerte correlación fuera de la diagonal; y la regla de negocio x ≥ 0 se avisa pero no se impone.');
    $('conclusiones-lista').innerHTML = items.map((i) => '<li>' + i + '</li>').join('');
  }

  /* ---------- Tarjetas estáticas ---------- */
  function buildCards() {
    $('var-cards').innerHTML = VARS.map((v, i) => '<article class="card var"><div class="ico">' + v.icono + '</div><h3>x' + sub(i) + ' · ' + esc(v.nombre) + '</h3><p>' + esc(v.desc) + '</p><small>Unidad: miles de bolivianos (Bs)</small></article>').join('');
    $('scen-cards').innerHTML = SCEN_KEYS.map((k) => {
      const s = SCENARIOS[k];
      return '<article class="card scen-card" data-key="' + k + '"><span class="badge ' + s.tono + '">' + (SCEN_KEYS.indexOf(k) + 1) + '. ' + s.nombre + '</span><p>' + esc(s.descripcion) + '</p><details><summary>Justificación de los datos</summary><ul>' + s.justificacion.map((j) => '<li>' + esc(j) + '</li>').join('') + '</ul></details><button class="btn" type="button" data-load="' + k + '">Cargar y ejecutar</button></article>';
    }).join('');
  }

  /* ===================================================================
     7. GRÁFICO DE CONVERGENCIA (canvas, escala semilogarítmica)
     =================================================================== */
  function drawChart() {
    const cv = $('chart');
    if (!cv || !cv.getContext) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const cssW = cv.clientWidth || (cv.parentElement && cv.parentElement.clientWidth) || 640;
    const cssH = Math.max(260, Math.min(420, Math.round(cssW * 0.55)));
    const dpr = global.devicePixelRatio || 1;
    cv.style.height = cssH + 'px'; cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, cssW, cssH);
    const cs = getComputedStyle(document.documentElement);
    const C = (n, f) => (cs.getPropertyValue(n).trim() || f);
    const text = C('--text', '#1b2433'); const muted = C('--muted', '#64708a'); const grid = C('--line', '#dfe5f0');
    const cGC = C('--gc', '#e07a1f'); const cGCP = C('--gcp', '#2f5bea');
    const res = state.lastResult;
    state.chart = null;
    if (!res || !res.ok) { ctx.fillStyle = muted; ctx.font = '14px system-ui, sans-serif'; ctx.fillText('Sin datos válidos para graficar.', 16, 32); return; }
    const gc = res.gc.history; const gcp = res.gcp.history; const tol = res.tol;
    const m = { l: 62, r: 16, t: 14, b: 44 };
    const w = cssW - m.l - m.r; const h = cssH - m.t - m.b;
    const ly = (v) => Math.log10(Math.max(v, 1e-17));
    const all = gc.concat(gcp, [tol]).map(ly);
    const yMin = Math.floor(Math.min.apply(null, all)); let yMax = Math.ceil(Math.max.apply(null, all) + 0.1);
    if (yMax <= yMin) yMax = yMin + 1;
    const kmax = Math.max(1, gc.length - 1, gcp.length - 1);
    const X = (k) => m.l + (k / kmax) * w; const Y = (v) => m.t + (1 - (ly(v) - yMin) / (yMax - yMin)) * h;
    ctx.font = '12px system-ui, sans-serif'; ctx.textBaseline = 'middle';
    const stepY = yMax - yMin > 12 ? 2 : 1;
    for (let e = yMin; e <= yMax; e += stepY) {
      const y = m.t + (1 - (e - yMin) / (yMax - yMin)) * h;
      ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + w, y); ctx.stroke();
      ctx.fillStyle = muted; ctx.textAlign = 'right'; ctx.fillText('10' + sup(e), m.l - 8, y);
    }
    const stepX = Math.max(1, Math.ceil(kmax / 10)); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let k = 0; k <= kmax; k += stepX) { ctx.fillStyle = muted; ctx.fillText(String(k), X(k), m.t + h + 6); }
    ctx.fillStyle = text; ctx.fillText('Iteración k', m.l + w / 2, m.t + h + 24);
    ctx.save(); ctx.translate(14, m.t + h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('‖rₖ‖₂ / ‖b‖₂  (escala log)', 0, 0); ctx.restore();
    // tolerancia
    ctx.setLineDash([6, 4]); ctx.strokeStyle = '#c0392b'; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(m.l, Y(tol)); ctx.lineTo(m.l + w, Y(tol)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#c0392b'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('tolerancia ' + sci(tol, 1), m.l + w - 4, Y(tol) - 2);
    const plot = (hst, color) => {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
      hst.forEach((v, k) => { if (k === 0) ctx.moveTo(X(k), Y(v)); else ctx.lineTo(X(k), Y(v)); }); ctx.stroke();
      if (kmax <= 60) hst.forEach((v, k) => { ctx.beginPath(); ctx.arc(X(k), Y(v), 3.2, 0, 2 * Math.PI); ctx.fill(); });
    };
    plot(gc, cGC); plot(gcp, cGCP);
    state.chart = { m, w, h, kmax, X, Y, gc, gcp };
  }
  function chartHover(e) {
    const cv = $('chart'); const tip = $('chart-tip'); const c = state.chart;
    if (!c || !tip) return;
    const rect = cv.getBoundingClientRect(); const px = e.clientX - rect.left;
    let k = Math.round(((px - c.m.l) / c.w) * c.kmax);
    k = Math.max(0, Math.min(c.kmax, k));
    tip.hidden = false;
    tip.innerHTML = '<strong>k = ' + k + '</strong><br><span class="dot gc"></span>GC: ' + (k < c.gc.length ? sci(c.gc[k], 3) : '—') + '<br><span class="dot gcp"></span>GCP: ' + (k < c.gcp.length ? sci(c.gcp[k], 3) : '—');
    tip.style.left = Math.min(rect.width - 150, Math.max(4, c.X(k) + 10)) + 'px'; tip.style.top = '8px';
  }

  /* ===================================================================
     8. MÓDULO QA — pruebas reales ejecutadas bajo demanda
     =================================================================== */
  const chk = (ok, texto) => ({ ok: !!ok, texto });
  const isHidden = (id) => $(id).hidden;

  const QA_TESTS = [
    {
      id: 'QA-1', nombre: 'Carga de escenarios',
      entrada: 'Cargar con los botones (misma ruta de la interfaz) los escenarios Ideal, Bajo estrés y Mal condicionado, con tolerancia 10⁻⁶ y x₀ = 0.',
      esperado: 'Los 3 sistemas son válidos (SPD); GC y GCP convergen; la solución de Cholesky coincide con eliminación gaussiana; el error de GCP cumple ‖x−x_ref‖/‖x_ref‖ ≤ κ(A)·residuo; Ideal con κ(A) < 10 y Mal condicionado con κ(A) ≥ 10³.',
      run() {
        const snap = snapshotEditor(); const checks = [];
        tolInput.value = '1e-6'; maxIterInput.value = '1000'; symChk.checked = true;
        SCEN_KEYS.forEach((k) => {
          const r = loadScenario(k); const nm = SCENARIOS[k].nombre;
          if (!r || !r.ok) { checks.push(chk(false, nm + ': no se pudo ejecutar')); return; }
          const sys = r.sys; const xg = gaussSolve(sys.A, sys.b);
          const dG = xg ? norm2(xg.map((v, i) => v - r.xref[i])) / norm2(xg) : Infinity;
          checks.push(chk(r.gc.converged && r.gcp.converged, nm + ': GC ' + r.gc.iterations + ' it., GCP ' + r.gcp.iterations + ' it., ambos convergen'));
          checks.push(chk(r.gcp.trueRes < r.tol, nm + ': residuo relativo GCP = ' + sci(r.gcp.trueRes) + ' < ' + sci(r.tol, 1)));
          checks.push(chk(dG <= 1e-8, nm + ': Cholesky vs Gauss (independiente): diferencia relativa ' + sci(dG)));
          checks.push(chk(r.gcp.relErr <= r.kappaA * r.gcp.trueRes * (1 + 1e-9) + 1e-14, nm + ': error relativo ' + sci(r.gcp.relErr) + ' ≤ κ·residuo = ' + sci(r.kappaA * r.gcp.trueRes)));
          if (k === 'ideal') checks.push(chk(r.kappaA < 10, 'Ideal: κ(A) = ' + num(r.kappaA, 3) + ' < 10'));
          if (k === 'mal') checks.push(chk(r.kappaA >= 1e3 && r.kappaM < r.kappaA, 'Mal condicionado: κ(A) = ' + num(r.kappaA, 3) + ' ≥ 10³ y κ(M⁻¹A) = ' + num(r.kappaM, 3) + ' < κ(A)'));
        });
        restoreEditor(snap); return checks;
      }
    },
    {
      id: 'QA-2', nombre: 'Validación de matriz no válida',
      entrada: 'Matrices deliberadamente inválidas: (a) no simétrica, (b) simétrica indefinida, (c) simétrica singular, (d) entrada no numérica, (e) tolerancia ≤ 0; además, edición real de un coeficiente en la interfaz con simetría automática desactivada.',
      esperado: 'Cada caso es rechazado con un mensaje claro, sin excepciones ni resultados numéricos, y la interfaz bloquea la ejecución.',
      run() {
        const checks = []; const base = () => scenarioSystem('ideal', 1e-6, 1000);
        const attempt = (nombre, mod, palabra) => {
          let r; let threw = false;
          try { const s = base(); mod(s); r = solveSystem(s, validateSystem(s)); } catch (e) { threw = true; }
          const ok = !threw && r && r.ok === false && r.errors.length > 0 && !r.xref && r.errors.join(' ').toLowerCase().indexOf(palabra) >= 0;
          checks.push(chk(ok, nombre + ': ' + (threw ? 'lanzó una excepción' : r && !r.ok ? 'rechazada → «' + r.errors[0] + '»' : 'NO fue rechazada')));
        };
        attempt('(a) No simétrica (a₁₂ = 5, a₂₁ = 2)', (s) => { s.A[0][1] = 5; }, 'simétrica');
        attempt('(b) Simétrica indefinida', (s) => { s.A = [[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]]; }, 'definida positiva');
        attempt('(c) Simétrica singular', (s) => { s.A = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]]; }, 'definida positiva');
        attempt('(d) Entrada no numérica', (s) => { s.A[2][2] = parseNumber('abc'); }, 'no numérica');
        attempt('(e) Tolerancia 0', (s) => { s.tol = 0; }, 'tolerancia');
        // Ruta real de la interfaz
        const snap = snapshotEditor();
        try {
          loadScenario('ideal'); symChk.checked = false;
          setInput('a-0-1', '5');
          const blocked = state.lastResult === null && !isHidden('editor-errores') && $('editor-errores').className.indexOf('error') >= 0 && document.querySelector('.js-result').hidden === true;
          checks.push(chk(blocked, 'Interfaz: tras editar a₁₂ = 5 sin simetría automática, ' + (blocked ? 'se mostró el error y se ocultaron los resultados' : 'la ejecución NO fue bloqueada')));
          setInput('a-0-1', '2');
          const back = state.lastResult && state.lastResult.ok;
          checks.push(chk(back, 'Interfaz: al corregir a₁₂ = 2 el sistema vuelve a calcularse'));
        } catch (e) { checks.push(chk(false, 'Interfaz: excepción ' + e.message)); }
        restoreEditor(snap); return checks;
      }
    },
    {
      id: 'QA-3', nombre: 'Edición dinámica',
      entrada: 'Desde la interfaz, partiendo del escenario Ideal: a₁₁ = 14, a₁₂ = 3 (con simetría automática, debe reflejarse en a₂₁) y b₂ = 16000.',
      esperado: 'Recalcula al instante: a₂₁ = 3; la solución cambia; la tabla en pantalla coincide con el estado; la referencia coincide con eliminación gaussiana independiente; GCP converge y los errores se recalculan.',
      run() {
        const snap = snapshotEditor(); const checks = [];
        tolInput.value = '1e-6'; maxIterInput.value = '1000'; symChk.checked = true;
        const antes = loadScenario('ideal'); const x0ref = antes.xref.slice();
        setInput('a-0-0', '14'); setInput('b-1', '16000'); setInput('a-0-1', '3');
        const r = state.lastResult;
        checks.push(chk($('a-1-0').value === '3', 'Simetría automática: a₂₁ = ' + $('a-1-0').value));
        checks.push(chk(r && r.ok, 'El sistema editado es válido y se recalculó'));
        if (r && r.ok) {
          const cambio = norm2(r.xref.map((v, i) => v - x0ref[i]));
          checks.push(chk(cambio > 1e-6, 'La solución de referencia cambió: ‖Δx‖ = ' + num(cambio, 4)));
          const sys = readSystem(); const xg = gaussSolve(sys.A, sys.b);
          const dG = norm2(xg.map((v, i) => v - r.xref[i])) / norm2(xg);
          checks.push(chk(dG <= 1e-9, 'Referencia vs Gauss independiente sobre los datos del editor: ' + sci(dG)));
          const cell = document.querySelector('[data-v="ref-0"]'); const cellG = document.querySelector('[data-v="gcp-1"]');
          checks.push(chk(cell && cell.textContent === num(r.xref[0], 6) && cellG && cellG.textContent === num(r.gcp.x[1], 6), 'La tabla en pantalla muestra x₁ = ' + (cell ? cell.textContent : '—') + ' y x₂(GCP) = ' + (cellG ? cellG.textContent : '—')));
          checks.push(chk(r.gcp.converged && r.gcp.trueRes < r.tol, 'GCP converge en ' + r.gcp.iterations + ' it., residuo ' + sci(r.gcp.trueRes)));
          const e = r.gcp.errors.every((q) => q.ea <= 1e-3 * Math.max(1, norm2(r.xref)));
          checks.push(chk(e, 'Errores absolutos por variable recalculados (máx. ' + sci(Math.max.apply(null, r.gcp.errors.map((q) => q.ea))) + ')'));
        }
        restoreEditor(snap); return checks;
      }
    },
    {
      id: 'QA-4', nombre: 'Comparación del precondicionamiento',
      entrada: 'Escenario Mal condicionado, tol = 10⁻⁶, x₀ = 0, máx. 1000 iteraciones: GC simple frente a GCP con Jacobi.',
      esperado: 'Ambos convergen y GCP alcanza la tolerancia en un número de iteraciones menor o igual que GC; además κ(M⁻¹A) < κ(A).',
      run() {
        const s = scenarioSystem('mal', 1e-6, 1000); const r = solveSystem(s, validateSystem(s)); const checks = [];
        checks.push(chk(r.ok && r.gc.converged && r.gcp.converged, 'Convergencia: GC ' + (r.gc.converged ? 'sí' : 'no') + ', GCP ' + (r.gcp.converged ? 'sí' : 'no')));
        checks.push(chk(r.gcp.iterations <= r.gc.iterations, 'Iteraciones: GCP = ' + r.gcp.iterations + ' ≤ GC = ' + r.gc.iterations));
        checks.push(chk(r.kappaM < r.kappaA, 'Condicionamiento: κ(M⁻¹A) = ' + num(r.kappaM, 3) + ' < κ(A) = ' + num(r.kappaA, 3)));
        return checks;
      }
    }
  ];
  const qaResults = {};

  function buildQA() {
    $('qa-body').innerHTML = QA_TESTS.map((t) => '<tr id="qa-row-' + t.id + '"><th scope="row">' + t.id + '<br><small>' + esc(t.nombre) + '</small></th><td>' + esc(t.entrada) + '</td><td>' + esc(t.esperado) + '</td><td class="qa-obt"><span class="muted">Sin ejecutar.</span></td><td class="qa-con"><span class="tag pend">Pendiente</span></td></tr>').join('');
  }
  function runQA(only) {
    const list = only ? QA_TESTS.filter((t) => t.id === only) : QA_TESTS;
    list.forEach((t) => {
      let checks;
      try { checks = t.run(); } catch (e) { checks = [chk(false, 'Excepción durante la prueba: ' + e.message)]; }
      const pass = checks.length > 0 && checks.every((c) => c.ok);
      qaResults[t.id] = { pass, checks };
      const row = $('qa-row-' + t.id);
      row.querySelector('.qa-obt').innerHTML = '<ul class="qa-list">' + checks.map((c) => '<li class="' + (c.ok ? 'pass' : 'fail') + '">' + (c.ok ? '✔ ' : '✘ ') + esc(c.texto) + '</li>').join('') + '</ul>';
      row.querySelector('.qa-con').innerHTML = pass ? '<span class="tag ok">Aprobada</span><br><small>' + checks.length + '/' + checks.length + ' verificaciones</small>' : '<span class="tag bad">Fallida</span><br><small>' + checks.filter((c) => c.ok).length + '/' + checks.length + ' verificaciones</small>';
    });
    const done = Object.keys(qaResults).length; const ok = Object.keys(qaResults).filter((k) => qaResults[k].pass).length;
    $('qa-resumen').textContent = done + ' de ' + QA_TESTS.length + ' pruebas ejecutadas · ' + ok + ' aprobadas · ' + (done - ok) + ' fallidas.';
    $('qa-resumen').className = 'badge ' + (done === ok ? 'ok' : 'bad');
  }

  /* ===================================================================
     9. DATOS DEL ESTUDIANTE (editables, se guardan en el navegador)
     =================================================================== */
  const STUDENT_FIELDS = ['nombre', 'carrera', 'universidad', 'mencion', 'materia', 'docente', 'fecha'];
  const STUDENT_DEFAULTS = { nombre: 'Javier Averanga Quelka', carrera: 'Informática', universidad: 'Universidad Mayor de San Andrés (UMSA)', mencion: 'Ingeniería de Sistemas', materia: 'Métodos Numéricos', docente: '', fecha: '' };
  function initStudent() {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem('gcp-estudiante') || '{}') || {}; } catch (e) { saved = {}; }
    const today = new Date().toISOString().slice(0, 10);
    STUDENT_FIELDS.forEach((f) => {
      const el = $('st-' + f); if (!el) return;
      el.value = saved[f] !== undefined ? saved[f] : (f === 'fecha' ? today : STUDENT_DEFAULTS[f]);
      el.addEventListener('input', () => {
        const o = {}; STUDENT_FIELDS.forEach((g) => { o[g] = $('st-' + g).value; });
        try { localStorage.setItem('gcp-estudiante', JSON.stringify(o)); } catch (e) { /* almacenamiento no disponible */ }
      });
    });
  }

  /* ===================================================================
     10. INICIALIZACIÓN
     =================================================================== */
  function init() {
    tolInput = $('tol'); maxIterInput = $('maxiter'); symChk = $('sym-auto');
    initStudent(); buildCards(); buildEditor(); buildQA();
    $('editor').addEventListener('input', onEditorInput);
    document.addEventListener('click', (e) => {
      const b = e.target.closest ? e.target.closest('[data-load]') : null;
      if (b) { loadScenario(b.dataset.load); renderComparativa(); const t = $('comparacion'); if (t && t.scrollIntoView && e.isTrusted) t.scrollIntoView({ behavior: 'smooth' }); }
    });
    $('btn-recalc').addEventListener('click', () => { recalc(); renderComparativa(); });
    $('btn-reset').addEventListener('click', () => { tolInput.value = '1e-6'; maxIterInput.value = '1000'; loadScenario(state.scenarioKey); renderComparativa(); });
    $('btn-todos').addEventListener('click', () => { renderComparativa(); const t = $('tabla-comparativa-host'); if (t && t.scrollIntoView) t.scrollIntoView({ behavior: 'smooth' }); });
    $('btn-qa').addEventListener('click', () => runQA());
    const cv = $('chart');
    if (cv) { cv.addEventListener('pointermove', chartHover); cv.addEventListener('pointerleave', () => { $('chart-tip').hidden = true; }); }
    global.addEventListener('resize', drawChart);
    tolInput.value = '1e-6'; maxIterInput.value = '1000';
    loadScenario('ideal'); renderComparativa();
  }

  global.MN = { parseNumber, cholesky, choleskySolve, gaussSolve, eigenvaluesSym, leadingMinors, conjugateGradient, validateSystem, solveSystem, SCENARIOS, scenarioSystem, state, runQA, qaResults, loadScenario, readSystem, num, sci, QA_TESTS };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})(typeof window !== 'undefined' ? window : this);
