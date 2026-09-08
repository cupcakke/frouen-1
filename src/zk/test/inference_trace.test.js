"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const assert = require("assert");
const snarkjs = require("snarkjs");
const { buildPoseidon } = require("circomlibjs");

const FIELD_MODULUS =
  21888242871839275222246405745257275088548364400416034343698204186575808495617n;

const SCALE_BITS = 20n;
const SCALE = 1n << SCALE_BITS;
const EXP_SCALE_BITS = 40n;
const EXP_SCALE = 1n << EXP_SCALE_BITS;
const EXP_GROUP = 3;
const CLIP_MIN_MAGNITUDE = 5n;
const CLIP_MAX = 5n;
const OFTB_SCALE_FIXED = 741455n;
const VALUE_LIMIT = 62914560000n;
const WEIGHT_LIMIT = 68685922304n;

const BATCH_SIZE = 2;
const NUM_LAYERS = 2;
const DIM = 16;
const HALF = DIM / 2;
const MODEL_DEPTH = 4;

const EXP_BIT_FACTOR = [
  1099512676353n,
  1099513724930n,
  1099515822088n,
  1099520016416n,
  1099528405120n,
  1099545182720n,
  1099578738688n,
  1099645853696n,
  1099780096003n,
  1100048629781n,
  1100585894059n,
  1101661209942n,
  1103814994613n,
  1108135204352n,
  1116826416478n,
  1134413873426n,
  1170424035283n,
  1245909900143n,
  1411800876008n,
  1812788208096n,
  2988782477963n,
  8124353099063n,
  60031300816501n,
  3277597968664119n,
  9770381843001049673n,
  86820692884470725459698192n,
];

const EXP_NEG_INT = [
  1099511627776n,
  404487723188n,
  148802717567n,
  54741460583n,
  20138257928n,
  7408451073n,
  2725416841n,
  1002624824n,
  368845060n,
  135690515n,
  49917751n,
  18363714n,
  6755633n,
  2485258n,
  914275n,
  336343n,
  123734n,
  45519n,
  16746n,
  6160n,
  2266n,
];

function toField(value) {
  const reduced = value % FIELD_MODULUS;
  return reduced < 0n ? reduced + FIELD_MODULUS : reduced;
}

function ceilLog2(value) {
  let bound = 1n;
  let result = 0;
  while (bound < value) {
    bound *= 2n;
    result += 1;
  }
  return result;
}

function shiftRight(value, bits) {
  const divisor = 1n << bits;
  let quotient = value / divisor;
  if (value < 0n && quotient * divisor !== value) {
    quotient -= 1n;
  }
  return quotient;
}

function clampToRange(value, lowerMagnitude, upper) {
  const lowerBounded = value < -lowerMagnitude ? -lowerMagnitude : value;
  return upper < lowerBounded ? upper : lowerBounded;
}

function fixedExp(value) {
  const span = (CLIP_MIN_MAGNITUDE + CLIP_MAX) * SCALE;
  const shifted = value + CLIP_MIN_MAGNITUDE * SCALE;
  assert.ok(shifted >= 0n && shifted <= span, "exponent argument outside the clip range");
  const bits = ceilLog2(span + 1n);
  let accumulator = EXP_NEG_INT[Number(CLIP_MIN_MAGNITUDE)];
  for (let group = 0; group * EXP_GROUP < bits; group += 1) {
    const used = Math.min(EXP_GROUP, bits - group * EXP_GROUP);
    let partial = accumulator;
    for (let offset = 0; offset < used; offset += 1) {
      const index = group * EXP_GROUP + offset;
      const bit = (shifted >> BigInt(index)) & 1n;
      partial *= bit === 1n ? EXP_BIT_FACTOR[index] : EXP_SCALE;
    }
    accumulator = partial >> (EXP_SCALE_BITS * BigInt(used));
  }
  return accumulator >> (EXP_SCALE_BITS - SCALE_BITS);
}

function layerForward(state, weightsS, weightsT) {
  const result = new Array(DIM);
  for (let i = 0; i < HALF; i += 1) {
    const x1 = state[i];
    const x2 = state[HALF + i];
    const preActivation = shiftRight(weightsS[i][0] * x2, SCALE_BITS) + weightsS[i][1];
    const clamped = clampToRange(preActivation, CLIP_MIN_MAGNITUDE * SCALE, CLIP_MAX * SCALE);
    const scale = fixedExp(clamped);
    const y1 = shiftRight(x1 * scale, SCALE_BITS);
    const translation = shiftRight(weightsT[i][0] * y1, SCALE_BITS) + weightsT[i][1];
    const y2 = x2 + translation;
    const low = shiftRight((y1 - y2) * OFTB_SCALE_FIXED, SCALE_BITS);
    const high = shiftRight((y1 + y2) * OFTB_SCALE_FIXED, SCALE_BITS);
    result[i] = clampToRange(low, VALUE_LIMIT, VALUE_LIMIT);
    result[HALF + i] = clampToRange(high, VALUE_LIMIT, VALUE_LIMIT);
  }
  return result;
}

function stackForward(tokens, weightsS, weightsT) {
  let state = tokens.slice();
  for (let layer = 0; layer < NUM_LAYERS; layer += 1) {
    state = layerForward(state, weightsS[layer], weightsT[layer]);
  }
  return state;
}

function makeRandomGenerator(seed) {
  let state = BigInt(seed) & 0xffffffffffffffffn;
  return () => {
    state = (state * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
    return Number((state >> 33n) & 0x7fffffffn);
  };
}

function randomFixed(random, magnitude) {
  const span = 2 * magnitude + 1;
  return BigInt(random() % span) - BigInt(magnitude);
}

function randomFieldElement(random) {
  let value = 0n;
  for (let i = 0; i < 8; i += 1) {
    value = (value << 31n) + BigInt(random());
  }
  return toField(value + 1n);
}

function buildHasher(poseidon) {
  const hash = (inputs) => poseidon.F.toObject(poseidon(inputs.map((value) => toField(value))));
  const chain = (values) => {
    const total = values.length;
    const chunks = Math.ceil(total / 6);
    let previous = 0n;
    for (let chunk = 0; chunk < chunks; chunk += 1) {
      const inputs = [];
      for (let j = 0; j < 6; j += 1) {
        const index = chunk * 6 + j;
        inputs.push(index < total ? toField(values[index]) : 0n);
      }
      inputs.push(previous);
      inputs.push(BigInt(total));
      previous = hash(inputs);
    }
    return previous;
  };
  const commit = (value, blinding) => hash([value, blinding]);
  const merkleRoot = (leaves) => {
    let level = leaves.slice();
    while (level.length > 1) {
      const next = [];
      for (let i = 0; i < level.length; i += 2) {
        next.push(hash([level[i], level[i + 1]]));
      }
      level = next;
    }
    return level[0];
  };
  const merklePathRoot = (leaf, elements, indices) => {
    let current = leaf;
    for (let level = 0; level < elements.length; level += 1) {
      const left = indices[level] === 1n ? elements[level] : current;
      const right = indices[level] === 1n ? current : elements[level];
      current = hash([left, right]);
    }
    return current;
  };
  return { hash, chain, commit, merkleRoot, merklePathRoot };
}

function buildWitnessInput(hasher, random, errorPerturbation, tokenMagnitude) {
  const weightsS = [];
  const weightsT = [];
  for (let layer = 0; layer < NUM_LAYERS; layer += 1) {
    const layerS = [];
    const layerT = [];
    for (let i = 0; i < HALF; i += 1) {
      layerS.push([randomFixed(random, 700000), randomFixed(random, 400000)]);
      layerT.push([randomFixed(random, 500000), randomFixed(random, 300000)]);
    }
    weightsS.push(layerS);
    weightsT.push(layerT);
  }

  const tokens = [];
  const outputs = [];
  const expectedOutputs = [];
  for (let b = 0; b < BATCH_SIZE; b += 1) {
    const sample = [];
    for (let i = 0; i < DIM; i += 1) {
      sample.push(randomFixed(random, tokenMagnitude));
    }
    const produced = stackForward(sample, weightsS, weightsT);
    const reference = produced.map((value, index) =>
      clampToRange(
        value + BigInt(((index % 5) - 2) * errorPerturbation),
        VALUE_LIMIT,
        VALUE_LIMIT,
      ),
    );
    tokens.push(sample);
    outputs.push(produced);
    expectedOutputs.push(reference);
  }

  const tokenBlindings = [];
  const outputBlindings = [];
  const referenceBlindings = [];
  for (let b = 0; b < BATCH_SIZE; b += 1) {
    tokenBlindings.push(randomFieldElement(random));
    outputBlindings.push(randomFieldElement(random));
    referenceBlindings.push(randomFieldElement(random));
  }

  const inputCommitments = [];
  const outputCommitments = [];
  const referenceCommitments = [];
  for (let b = 0; b < BATCH_SIZE; b += 1) {
    inputCommitments.push(hasher.commit(hasher.chain(tokens[b]), tokenBlindings[b]));
    outputCommitments.push(hasher.commit(hasher.chain(outputs[b]), outputBlindings[b]));
    referenceCommitments.push(
      hasher.commit(hasher.chain(expectedOutputs[b]), referenceBlindings[b]),
    );
  }

  const flatWeights = [];
  for (let layer = 0; layer < NUM_LAYERS; layer += 1) {
    for (let i = 0; i < HALF; i += 1) {
      flatWeights.push(weightsS[layer][i][0]);
      flatWeights.push(weightsS[layer][i][1]);
      flatWeights.push(weightsT[layer][i][0]);
      flatWeights.push(weightsT[layer][i][1]);
    }
  }

  const weightsBlinding = randomFieldElement(random);
  const modelLeaf = hasher.commit(hasher.chain(flatWeights), weightsBlinding);
  const modelPathElements = [];
  const modelPathIndices = [];
  for (let level = 0; level < MODEL_DEPTH; level += 1) {
    modelPathElements.push(randomFieldElement(random));
    modelPathIndices.push(BigInt(random() % 2));
  }
  const modelRoot = hasher.merklePathRoot(modelLeaf, modelPathElements, modelPathIndices);

  let maximumAbsoluteError = 0n;
  let maximumSquaredError = 0n;
  for (let b = 0; b < BATCH_SIZE; b += 1) {
    let squared = 0n;
    for (let i = 0; i < DIM; i += 1) {
      const difference = outputs[b][i] - expectedOutputs[b][i];
      const absolute = difference < 0n ? -difference : difference;
      if (absolute > maximumAbsoluteError) {
        maximumAbsoluteError = absolute;
      }
      squared += absolute * absolute;
    }
    if (squared > maximumSquaredError) {
      maximumSquaredError = squared;
    }
  }

  const input = {
    tokens: tokens.map((sample) => sample.map((value) => toField(value).toString())),
    token_blindings: tokenBlindings.map((value) => value.toString()),
    output_blindings: outputBlindings.map((value) => value.toString()),
    reference_blindings: referenceBlindings.map((value) => value.toString()),
    expected_outputs: expectedOutputs.map((sample) =>
      sample.map((value) => toField(value).toString()),
    ),
    weights_s: weightsS.map((layer) =>
      layer.map((pair) => pair.map((value) => toField(value).toString())),
    ),
    weights_t: weightsT.map((layer) =>
      layer.map((pair) => pair.map((value) => toField(value).toString())),
    ),
    weights_blinding: weightsBlinding.toString(),
    model_path_elements: modelPathElements.map((value) => value.toString()),
    model_path_indices: modelPathIndices.map((value) => value.toString()),
    input_root: hasher.merkleRoot(inputCommitments).toString(),
    output_root: hasher.merkleRoot(outputCommitments).toString(),
    reference_root: hasher.merkleRoot(referenceCommitments).toString(),
    model_root: modelRoot.toString(),
    max_error_squared: maximumSquaredError.toString(),
    max_abs_error: maximumAbsoluteError.toString(),
  };

  return { input, weightsS, weightsT, tokens, outputs };
}

async function calculateWitness(buildDir, input, witnessPath) {
  const builder = require(path.join(buildDir, "inference_trace_js", "witness_calculator.js"));
  const code = fs.readFileSync(path.join(buildDir, "inference_trace_js", "inference_trace.wasm"));
  const calculator = await builder(code);
  const buffer = await calculator.calculateWTNSBin(input, 0);
  fs.writeFileSync(witnessPath, buffer);
}

async function expectWitnessFailure(buildDir, input, witnessPath, description) {
  let failed = false;
  try {
    await calculateWitness(buildDir, input, witnessPath);
  } catch (error) {
    failed = true;
  }
  assert.ok(failed, description);
}

async function main() {
  const buildDir = process.env.JAIDE_ZK_DIR || path.join(__dirname, "..");
  const r1csPath = path.join(buildDir, "inference_trace.r1cs");
  assert.ok(
    fs.existsSync(r1csPath),
    `hiányzó R1CS: ${r1csPath}; futtasd előbb a zig build zk -Dzk=true lépést`,
  );

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "jaide-zk-"));
  const witnessPath = path.join(workDir, "witness.wtns");

  const poseidon = await buildPoseidon();
  const hasher = buildHasher(poseidon);
  const random = makeRandomGenerator(20240917);

  const scenarios = [
    { seed: 20240917, perturbation: 3, magnitude: 3 * Number(SCALE) },
    { seed: 1337, perturbation: 1, magnitude: Number(SCALE) / 64 },
    { seed: 424242, perturbation: 11, magnitude: Number(VALUE_LIMIT) },
    { seed: 8675309, perturbation: 0, magnitude: 900 * Number(SCALE) },
  ];

  for (const scenario of scenarios) {
    const generator = makeRandomGenerator(scenario.seed);
    const candidate = buildWitnessInput(
      hasher,
      generator,
      scenario.perturbation,
      scenario.magnitude,
    );
    await calculateWitness(buildDir, candidate.input, witnessPath);
    await snarkjs.wtns.check(r1csPath, witnessPath, undefined);
  }

  const valid = buildWitnessInput(hasher, random, 3, 3 * Number(SCALE));
  await calculateWitness(buildDir, valid.input, witnessPath);
  await snarkjs.wtns.check(r1csPath, witnessPath, undefined);

  const tamperedRoot = { ...valid.input };
  tamperedRoot.output_root = toField(BigInt(tamperedRoot.output_root) + 1n).toString();
  await expectWitnessFailure(
    buildDir,
    tamperedRoot,
    witnessPath,
    "a meghamisított kimeneti gyökér nem okozott hibát",
  );

  const tamperedReference = { ...valid.input };
  tamperedReference.reference_root = toField(
    BigInt(tamperedReference.reference_root) + 1n,
  ).toString();
  await expectWitnessFailure(
    buildDir,
    tamperedReference,
    witnessPath,
    "a meghamisított referencia gyökér nem okozott hibát",
  );

  const tamperedModel = { ...valid.input };
  tamperedModel.model_root = toField(BigInt(tamperedModel.model_root) + 1n).toString();
  await expectWitnessFailure(
    buildDir,
    tamperedModel,
    witnessPath,
    "a meghamisított modell gyökér nem okozott hibát",
  );

  const tightBound = { ...valid.input };
  tightBound.max_abs_error = (BigInt(valid.input.max_abs_error) - 1n).toString();
  await expectWitnessFailure(
    buildDir,
    tightBound,
    witnessPath,
    "a túl szoros abszolút hibakorlát nem okozott hibát",
  );

  const tightSquaredBound = { ...valid.input };
  tightSquaredBound.max_error_squared = (
    BigInt(valid.input.max_error_squared) - 1n
  ).toString();
  await expectWitnessFailure(
    buildDir,
    tightSquaredBound,
    witnessPath,
    "a túl szoros négyzetes hibakorlát nem okozott hibát",
  );

  const overflowTokens = buildWitnessInput(
    hasher,
    makeRandomGenerator(7411),
    3,
    3 * Number(SCALE),
  );
  overflowTokens.input.tokens[0][0] = toField(VALUE_LIMIT + 1n).toString();
  await expectWitnessFailure(
    buildDir,
    overflowTokens.input,
    witnessPath,
    "a tartományon kívüli token nem okozott hibát",
  );

  const overflowWeights = buildWitnessInput(
    hasher,
    makeRandomGenerator(9931),
    3,
    3 * Number(SCALE),
  );
  overflowWeights.input.weights_s[0][0][0] = toField(WEIGHT_LIMIT + 1n).toString();
  await expectWitnessFailure(
    buildDir,
    overflowWeights.input,
    witnessPath,
    "a tartományon kívüli súly nem okozott hibát",
  );

  fs.rmSync(workDir, { recursive: true, force: true });

  process.stdout.write("inference_trace: minden ellenőrzés sikeres\n");
}

main()
  .then(() => {
    process.exit(0);
  })
  .catch((error) => {
    process.stderr.write(`${error.stack || error}\n`);
    process.exit(1);
  });
