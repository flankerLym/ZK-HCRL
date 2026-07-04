pragma circom 2.1.6;

// Self-contained comparison utilities for ZK-VOS ablation benchmarks.
// All values are expected to be small non-negative scaled integers.

template Num2Bits(n) {
    signal input in;
    signal output out[n];
    var lc = 0;
    for (var i = 0; i < n; i++) {
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] - 1) === 0;
        lc += out[i] * (1 << i);
    }
    lc === in;
}

template LessEqThan(n) {
    signal input a;
    signal input b;
    signal output out;
    component bits = Num2Bits(n + 1);
    // out = 1 iff a <= b, for a,b in [0, 2^n - 1].
    bits.in <== a + (1 << n) - b - 1;
    out <== 1 - bits.out[n];
}

template And2() {
    signal input a;
    signal input b;
    signal output out;
    a * (a - 1) === 0;
    b * (b - 1) === 0;
    out <== a * b;
}

template And3() {
    signal input a;
    signal input b;
    signal input c;
    signal output out;
    component ab = And2();
    ab.a <== a;
    ab.b <== b;
    component abc = And2();
    abc.a <== ab.out;
    abc.b <== c;
    out <== abc.out;
}

template RiskAware() {
    signal input requestId;
    signal input selectedOracleId;
    signal input serviceMatch;
    signal input cooldownFlag;
    signal input repEff;
    signal input cost;
    signal input risk;
    signal input latencyEst;
    signal input reputationThreshold;
    signal input costBudget;
    signal input riskBudget;
    signal input deadline;
    signal output zkIsValid;

    serviceMatch * (serviceMatch - 1) === 0;
    cooldownFlag * (cooldownFlag - 1) === 0;
    signal noCooldown;
    noCooldown <== 1 - cooldownFlag;

    component repOk = LessEqThan(32);
    repOk.a <== reputationThreshold;
    repOk.b <== repEff;

    component costOk = LessEqThan(32);
    costOk.a <== cost;
    costOk.b <== costBudget;

    component riskOk = LessEqThan(32);
    riskOk.a <== risk;
    riskOk.b <== riskBudget;

    component latencyOk = LessEqThan(32);
    latencyOk.a <== latencyEst;
    latencyOk.b <== deadline;

    component a = And3();
    a.a <== serviceMatch;
    a.b <== noCooldown;
    a.c <== repOk.out;

    component b = And3();
    b.a <== costOk.out;
    b.b <== riskOk.out;
    b.c <== latencyOk.out;

    component allOk = And2();
    allOk.a <== a.out;
    allOk.b <== b.out;
    zkIsValid <== allOk.out;
    zkIsValid === 1;
}

component main { public [requestId, selectedOracleId] } = RiskAware();
