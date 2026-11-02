//! Decision-model runtime demo: a tiny MLP (6 -> 16 -> 3, approve / review /
//! reject for card transactions) trained outside Rul and exported to a `.rls`
//! graph that runs on the `MatVec` + `Exp` opcodes.
//!
//! Usage: `cargo run --release --example decision_model [dir]`
//! where `dir` defaults to `examples/decision` and must contain `model.rls`
//! plus `data/arena_{1..4}.bin`, `data/samples.bin`, `data/labels.bin`, and
//! `data/probs.bin` (numpy reference probabilities). Regenerate with
//! `python3 examples/decision/generate.py`.
//!
//! The runner scores every held-out sample through Tier 2, cross-checks the
//! first samples against Tier 0 and the numpy reference probabilities, and
//! reports accuracy and per-decision latency.

use rul::bytecode::Opcode;
use rul::engine::{execute_tier0_interpreter, execute_tier2_fused, trace_program, RuntimeValue};
use rul::transpiler::assemble_rls;
use std::time::Instant;

fn read_f64s(path: &std::path::Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(8),
        "{} not f64-aligned",
        path.display()
    );
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/decision".to_string());
    let dir = std::path::Path::new(&dir);
    let data = dir.join("data");

    let src = std::fs::read_to_string(dir.join("model.rls")).expect("model.rls");
    let program = assemble_rls(&src).expect("assemble model.rls");
    let summary = rul::verifier::verify_program(&program).expect("verify model.rls");
    println!(
        "model verified: {} nodes, output shape {:?}",
        summary.node_count, summary.output_shape
    );

    let w1 = read_f64s(&data.join("arena_1.bin"));
    let b1 = read_f64s(&data.join("arena_2.bin"));
    let w2 = read_f64s(&data.join("arena_3.bin"));
    let b2 = read_f64s(&data.join("arena_4.bin"));
    let samples = read_f64s(&data.join("samples.bin"));
    let labels = read_f64s(&data.join("labels.bin"));
    let ref_probs = read_f64s(&data.join("probs.bin"));
    let n = labels.len();
    assert_eq!(samples.len(), n * 6);
    assert_eq!(ref_probs.len(), n * 3);

    // Locate the softmax probability node (the Div feeding the final ReduceMax).
    let prob_node = program
        .nodes
        .iter()
        .rposition(|f| f.opcode == Opcode::Div)
        .expect("softmax Div node");
    let class_names = ["approve", "review", "reject"];

    let mut correct = 0usize;
    let mut tier_agree = 0usize;
    let mut max_prob_diff = 0.0f64;
    let mut decisions = [0usize; 3];
    let t_start = Instant::now();
    for i in 0..n {
        let x = &samples[i * 6..(i + 1) * 6];
        let inputs: Vec<&[f64]> = vec![x, &w1, &b1, &w2, &b2];
        let trace = trace_program(&program, &inputs).expect("trace");
        let probs = &trace.entries[prob_node].sample_head;
        assert_eq!(probs.len(), 3, "probability node must yield 3 samples");
        let pred = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k)
            .unwrap();
        decisions[pred] += 1;
        if (pred as f64 - labels[i]).abs() < 0.5 {
            correct += 1;
        }
        for (k, &p) in probs.iter().enumerate() {
            max_prob_diff = max_prob_diff.max((p - ref_probs[i * 3 + k]).abs());
        }
        // Cross-check Tier 0 vs Tier 2 final confidence on the first 200 samples.
        if i < 200 {
            let t0 = execute_tier0_interpreter(&program, &inputs).expect("tier0");
            let t2 = execute_tier2_fused(&program, &inputs).expect("tier2");
            if let (RuntimeValue::Scalar(a), RuntimeValue::Scalar(b)) = (t0.output, t2.output) {
                if (a - b).abs() < 1e-12 {
                    tier_agree += 1;
                }
            }
        }
    }
    let total_us = t_start.elapsed().as_micros() as f64;

    // Pure Tier-2 scoring latency (no trace overhead), 3 passes over the set.
    let t_start = Instant::now();
    for _ in 0..3 {
        for i in 0..n {
            let x = &samples[i * 6..(i + 1) * 6];
            let inputs: Vec<&[f64]> = vec![x, &w1, &b1, &w2, &b2];
            let rep = execute_tier2_fused(&program, &inputs).expect("tier2");
            std::hint::black_box(rep.output);
        }
    }
    let tier2_us_per = t_start.elapsed().as_micros() as f64 / (3.0 * n as f64);

    println!("samples: {n}");
    println!(
        "rul accuracy: {:.4} ({correct}/{n})",
        correct as f64 / n as f64
    );
    println!(
        "decisions: approve={} review={} reject={}",
        decisions[0], decisions[1], decisions[2]
    );
    let _ = class_names;
    println!("max |p_rul - p_numpy| over all samples/classes: {max_prob_diff:.3e}");
    println!("tier0 vs tier2 confidence agreement (first 200): {tier_agree}/200");
    println!(
        "trace+score wall time per sample: {:.1} us",
        total_us / n as f64
    );
    println!("tier2 score-only latency per sample: {tier2_us_per:.2} us");
}
