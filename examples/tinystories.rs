//! TinyStories-8M (GPT-Neo, roneneldan) running on the Rul fork runtime.
//!
//! Usage: `cargo run --release --example tinystories [export_dir] [n_gen]`
//! `export_dir` (default `tinystories_export`) must contain `weights.bin`,
//! `manifest.txt`, `step0_logits.bin`, and `reference.json` as produced by
//! the lab exporter (`~/workspace/rul-lab/tinystories/export_weights.py`).
//!
//! Contract: the host owns the token loop and the KV cache. Per token and
//! layer it runs two verified Rul graphs - a LayerNorm graph, then a block
//! graph (attention over the cache + output projection + FFN with gelu_new
//! composed from `Exp`) - and one final graph (ln_f + tied-embedding
//! classifier). The host computes only the k/v projections for the new
//! token (the values that must be extracted into the cache) and the
//! embedding-row lookup; every other tensor op runs inside Rul graphs.
//! Local-attention layers (window 256) need no window logic because the
//! whole context here stays below 256 tokens.

use rul::engine::{execute_tier1_stencil, RuntimeValue};
use rul::transpiler::assemble_rls;
use std::collections::HashMap;
use std::time::Instant;

const D: usize = 256;
const L: usize = 8;
const H: usize = 16;
const DH: usize = 16;
const EPS: f64 = 1e-5;

struct Weights {
    blob: Vec<f64>,
    idx: HashMap<String, (usize, usize)>,
}

impl Weights {
    fn t(&self, name: &str) -> &[f64] {
        let (off, len) = self.idx[name];
        &self.blob[off..off + len]
    }
}

fn read_f64s(path: &std::path::Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(bytes.len().is_multiple_of(8));
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn matvec(w: &[f64], x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0;
        for (a, b) in row.iter().zip(x.iter()) {
            acc += a * b;
        }
        out[r] = acc;
    }
    out
}

fn run_column(program: &rul::bytecode::RulProgram, inputs: &[&[f64]]) -> Vec<f64> {
    let rep = execute_tier1_stencil(program, inputs).expect("execute");
    match rep.output {
        RuntimeValue::ColumnF64(c) => c,
        other => panic!("expected column output, got {other:?}"),
    }
}

/// LayerNorm graph: arenas 0=x, 1=gamma, 2=beta -> y (d,).
fn ln_rls() -> String {
    let mut s = String::from("cap 0x0\n");
    for a in 0..3 {
        s += &format!("arena {a} len={D} lane=f64x8 mut=false align=64 ver=1\n");
    }
    s += "node $x = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(0,1) a=0 b=0\n";
    s += "node $g = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(1,1) a=1 b=0\n";
    s += "node $b = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(2,1) a=2 b=0\n";
    s += &format!("node $dc = ConstF64 f64x1 fg=1 pol=simd proof=0 tok=pure const={D}.0\n");
    s += &format!("node $eps = ConstF64 f64x1 fg=1 pol=simd proof=0 tok=pure const={EPS:e}\n");
    s += "node $sum = ReduceSum f64x1 fg=1 pol=simd proof=0 tok=pure a=$x b=0\n";
    s += "node $mean = Div f64x1 fg=1 pol=simd proof=0 tok=pure a=$sum b=$dc\n";
    s += "node $xc = Sub f64x8 fg=1 pol=simd proof=0 tok=pure a=$x b=$mean\n";
    s += "node $sq = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$xc b=$xc\n";
    s += "node $vs = ReduceSum f64x1 fg=1 pol=simd proof=0 tok=pure a=$sq b=0\n";
    s += "node $var = Div f64x1 fg=1 pol=simd proof=0 tok=pure a=$vs b=$dc\n";
    s += "node $ve = Add f64x1 fg=1 pol=simd proof=0 tok=pure a=$var b=$eps\n";
    s += "node $r = Rsqrt f64x1 fg=1 pol=simd proof=0 tok=pure a=$ve b=0\n";
    s += "node $xn = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$xc b=$r\n";
    s += "node $yg = Mul f64x8 fg=1 pol=simd proof=0 tok=pure a=$xn b=$g\n";
    s += "node $y = Add f64x8 fg=1 pol=simd proof=0 tok=pure a=$yg b=$b\n";
    s += "output $y\n";
    s
}

/// Block graph for one layer at context length t (cache already includes
/// the current token's k/v). Arena layout is fixed by construction order;
/// the caller passes inputs in exactly the same order.
fn block_rls(t: usize) -> String {
    let mut s = String::from("cap 0x0\n");
    let mut arena = 0usize;
    let mut decl = |s: &mut String, len: usize| {
        *s += &format!("arena {arena} len={len} lane=f64x8 mut=false align=64 ver=1\n");
        arena += 1;
    };
    decl(&mut s, D); // 0: x
    decl(&mut s, D); // 1: y (ln1 output)
    for _ in 0..H {
        decl(&mut s, t * DH); // K per head
    }
    for _ in 0..H {
        decl(&mut s, DH * t); // V^T per head
    }
    for _ in 0..H {
        decl(&mut s, DH * D); // Wq per head
    }
    for _ in 0..H {
        decl(&mut s, D * DH); // Wo per head
    }
    decl(&mut s, D); // bo
    decl(&mut s, D); // ln2 g
    decl(&mut s, D); // ln2 b
    decl(&mut s, 1024 * D); // wfc
    decl(&mut s, 1024); // bfc
    decl(&mut s, D * 1024); // wcp
    decl(&mut s, D); // bcp

    let input = |s: &mut String, name: &str, a: usize| {
        *s += &format!(
            "node {name} = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read({a},1) a={a} b=0\n"
        );
    };
    input(&mut s, "$x", 0);
    input(&mut s, "$y", 1);
    let mut a = 2;
    for h in 0..H {
        input(&mut s, &format!("$k{h}"), a);
        a += 1;
    }
    for h in 0..H {
        input(&mut s, &format!("$vt{h}"), a);
        a += 1;
    }
    for h in 0..H {
        input(&mut s, &format!("$wq{h}"), a);
        a += 1;
    }
    for h in 0..H {
        input(&mut s, &format!("$wo{h}"), a);
        a += 1;
    }
    for (name, _) in [
        ("$bo", 0),
        ("$l2g", 0),
        ("$l2b", 0),
        ("$wfc", 0),
        ("$bfc", 0),
        ("$wcp", 0),
        ("$bcp", 0),
    ] {
        input(&mut s, name, a);
        a += 1;
    }
    let konst = |s: &mut String, name: &str, v: f64| {
        *s += &format!("node {name} = ConstF64 f64x1 fg=1 pol=simd proof=0 tok=pure const={v}\n");
    };
    konst(&mut s, "$dc", 256.0);
    konst(&mut s, "$eps", EPS);
    konst(&mut s, "$scale", 4.0);
    konst(&mut s, "$one", 1.0);
    konst(&mut s, "$two", 2.0);
    konst(&mut s, "$half", 0.5);
    konst(&mut s, "$c044", 0.044715);
    konst(&mut s, "$c0797", 0.7978845608028654);
    let node = |s: &mut String, name: &str, op: &str, a: &str, b: &str| {
        *s += &format!("node {name} = {op} f64x8 fg=1 pol=simd proof=0 tok=pure a={a} b={b}\n");
    };
    // attention
    for h in 0..H {
        node(
            &mut s,
            &format!("$q{h}"),
            "MatVec",
            &format!("$wq{h}"),
            "$y",
        );
        node(
            &mut s,
            &format!("$sc{h}"),
            "MatVec",
            &format!("$k{h}"),
            &format!("$q{h}"),
        );
        node(
            &mut s,
            &format!("$ss{h}"),
            "Div",
            &format!("$sc{h}"),
            "$scale",
        );
        node(
            &mut s,
            &format!("$mx{h}"),
            "ReduceMax",
            &format!("$ss{h}"),
            "0",
        );
        node(
            &mut s,
            &format!("$se{h}"),
            "Sub",
            &format!("$ss{h}"),
            &format!("$mx{h}"),
        );
        node(&mut s, &format!("$e{h}"), "Exp", &format!("$se{h}"), "0");
        node(
            &mut s,
            &format!("$es{h}"),
            "ReduceSum",
            &format!("$e{h}"),
            "0",
        );
        node(
            &mut s,
            &format!("$at{h}"),
            "Div",
            &format!("$e{h}"),
            &format!("$es{h}"),
        );
        node(
            &mut s,
            &format!("$o{h}"),
            "MatVec",
            &format!("$vt{h}"),
            &format!("$at{h}"),
        );
        node(
            &mut s,
            &format!("$p{h}"),
            "MatVec",
            &format!("$wo{h}"),
            &format!("$o{h}"),
        );
    }
    let mut acc = "$p0".to_string();
    for h in 1..H {
        let name = format!("$pa{h}");
        node(&mut s, &name, "Add", &acc, &format!("$p{h}"));
        acc = name;
    }
    node(&mut s, "$attn", "Add", &acc, "$bo");
    node(&mut s, "$x1", "Add", "$x", "$attn");
    // ln2
    node(&mut s, "$sum2", "ReduceSum", "$x1", "0");
    node(&mut s, "$mean2", "Div", "$sum2", "$dc");
    node(&mut s, "$xc2", "Sub", "$x1", "$mean2");
    node(&mut s, "$sq2", "Mul", "$xc2", "$xc2");
    node(&mut s, "$vs2", "ReduceSum", "$sq2", "0");
    node(&mut s, "$var2", "Div", "$vs2", "$dc");
    node(&mut s, "$ve2", "Add", "$var2", "$eps");
    node(&mut s, "$r2", "Rsqrt", "$ve2", "0");
    node(&mut s, "$xn2", "Mul", "$xc2", "$r2");
    node(&mut s, "$yg2", "Mul", "$xn2", "$l2g");
    node(&mut s, "$z", "Add", "$yg2", "$l2b");
    // ffn with gelu_new composed from Exp (tanh form)
    node(&mut s, "$f", "MatVec", "$wfc", "$z");
    node(&mut s, "$fb", "Add", "$f", "$bfc");
    node(&mut s, "$f2", "Mul", "$fb", "$fb");
    node(&mut s, "$f3", "Mul", "$f2", "$fb");
    node(&mut s, "$t1", "Mul", "$f3", "$c044");
    node(&mut s, "$u", "Add", "$fb", "$t1");
    node(&mut s, "$w", "Mul", "$u", "$c0797");
    node(&mut s, "$w2", "Mul", "$w", "$two");
    node(&mut s, "$ew", "Exp", "$w2", "0");
    node(&mut s, "$ed", "Add", "$ew", "$one");
    node(&mut s, "$td", "Div", "$two", "$ed");
    node(&mut s, "$th", "Sub", "$one", "$td");
    node(&mut s, "$tp", "Add", "$one", "$th");
    node(&mut s, "$g1", "Mul", "$fb", "$tp");
    node(&mut s, "$g", "Mul", "$g1", "$half");
    node(&mut s, "$m", "MatVec", "$wcp", "$g");
    node(&mut s, "$mb", "Add", "$m", "$bcp");
    node(&mut s, "$out", "Add", "$x1", "$mb");
    s += "output $out\n";
    s
}

/// Final graph: ln_f + tied-embedding classifier. Arenas 0=x, 1=g, 2=b, 3=wte.
fn final_rls() -> String {
    let mut s = ln_rls().replace("output $y\n", "");
    // Insert the wte arena declaration after arenas 0..2 (declaration order
    // defines arena ids) and append the classifier node.
    let wte_decl = format!(
        "arena 3 len={} lane=f64x8 mut=false align=64 ver=1\n",
        50257 * D
    );
    s = s.replacen(
        "node $x = ArenaInput",
        &(wte_decl + "node $x = ArenaInput"),
        1,
    );
    s += "node $wte = ArenaInput f64x8 fg=1 pol=simd proof=0 tok=read(3,1) a=3 b=0\n";
    s += "node $logits = MatVec f64x8 fg=1 pol=simd proof=0 tok=pure a=$wte b=$y\n";
    s += "output $logits\n";
    s
}

fn extract_ids(json: &str, key: &str) -> Vec<u32> {
    let start = json.find(key).unwrap_or_else(|| panic!("{key} missing"));
    let open = json[start..].find('[').unwrap() + start;
    let close = json[open..].find(']').unwrap() + open;
    json[open + 1..close]
        .split(',')
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .collect()
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tinystories_export".to_string());
    let n_gen: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let dir = std::path::Path::new(&dir);

    let blob = read_f64s(&dir.join("weights.bin"));
    let manifest = std::fs::read_to_string(dir.join("manifest.txt")).expect("manifest");
    let mut idx = HashMap::new();
    for line in manifest.lines() {
        let mut it = line.split_whitespace();
        let name = it.next().unwrap().to_string();
        let off: usize = it.next().unwrap().parse().unwrap();
        let len: usize = it.next().unwrap().parse().unwrap();
        idx.insert(name, (off, len));
    }
    let w = Weights { blob, idx };
    let ref_json = std::fs::read_to_string(dir.join("reference.json")).expect("reference.json");
    let prompt_ids = extract_ids(&ref_json, "\"prompt_ids\"");
    let ref_gen = extract_ids(&ref_json, "\"greedy_ids_40\"");
    let step0_ref = read_f64s(&dir.join("step0_logits.bin"));
    println!(
        "loaded {} tensors, prompt ids {:?}",
        w.idx.len(),
        prompt_ids
    );

    let ln_prog = assemble_rls(&ln_rls()).expect("assemble ln");
    let final_prog = assemble_rls(&final_rls()).expect("assemble final");

    // KV caches: k[l][h] = positions x DH (row-major), v[l][h] = positions x DH
    let mut cache_k: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); H]; L];
    let mut cache_v: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); H]; L];

    let process_token = |pos: usize,
                         token: u32,
                         w: &Weights,
                         cache_k: &mut Vec<Vec<Vec<f64>>>,
                         cache_v: &mut Vec<Vec<Vec<f64>>>|
     -> Vec<f64> {
        let wte = w.t("wte");
        let wpe = w.t("wpe");
        let mut x: Vec<f64> = (0..D)
            .map(|i| wte[token as usize * D + i] + wpe[pos * D + i])
            .collect();
        let t = pos + 1;
        for l in 0..L {
            // LN graph (ln_1 of this layer)
            let y = run_column(
                &ln_prog,
                &[
                    x.as_slice(),
                    w.t(&format!("l{l}_ln1_g")),
                    w.t(&format!("l{l}_ln1_b")),
                ],
            );
            // host: project k/v for the new token and append to the cache
            for h in 0..H {
                let kn = matvec(w.t(&format!("l{l}_k{h}")), &y, DH, D);
                let vn = matvec(w.t(&format!("l{l}_v{h}")), &y, DH, D);
                cache_k[l][h].extend_from_slice(&kn);
                cache_v[l][h].extend_from_slice(&vn);
            }
            // block graph at context length t
            let prog = assemble_rls(&block_rls(t)).expect("assemble block");
            let mut inputs: Vec<&[f64]> = vec![x.as_slice(), y.as_slice()];
            for kh in &cache_k[l] {
                inputs.push(kh.as_slice());
            }
            // V^T buffers: vt[j*t + i] = v_i[j]
            let mut vt_bufs: Vec<Vec<f64>> = Vec::with_capacity(H);
            for v in &cache_v[l] {
                let mut vt = vec![0.0f64; DH * t];
                for i in 0..t {
                    for j in 0..DH {
                        vt[j * t + i] = v[i * DH + j];
                    }
                }
                vt_bufs.push(vt);
            }
            for vt in &vt_bufs {
                inputs.push(vt.as_slice());
            }
            for h in 0..H {
                inputs.push(w.t(&format!("l{l}_q{h}")));
            }
            for h in 0..H {
                inputs.push(w.t(&format!("l{l}_wo{h}")));
            }
            inputs.push(w.t(&format!("l{l}_bo")));
            inputs.push(w.t(&format!("l{l}_ln2_g")));
            inputs.push(w.t(&format!("l{l}_ln2_b")));
            inputs.push(w.t(&format!("l{l}_wfc")));
            inputs.push(w.t(&format!("l{l}_bfc")));
            inputs.push(w.t(&format!("l{l}_wcp")));
            inputs.push(w.t(&format!("l{l}_bcp")));
            x = run_column(&prog, &inputs);
        }
        run_column(
            &final_prog,
            &[x.as_slice(), w.t("lnf_g"), w.t("lnf_b"), w.t("wte")],
        )
    };

    let t_start = Instant::now();
    let mut logits: Vec<f64> = Vec::new();
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        logits = process_token(pos, tok, &w, &mut cache_k, &mut cache_v);
    }
    let prompt_time = t_start.elapsed();

    // verify step-0 logits against the numpy reference
    let max_diff = logits
        .iter()
        .zip(step0_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let argmax = |v: &[f64]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap()
    };
    println!(
        "step0 logits: max |rul - numpy| = {max_diff:.3e}; argmax rul {} vs numpy {}",
        argmax(&logits),
        argmax(&step0_ref)
    );
    println!(
        "prompt processing: {:?} for {} tokens",
        prompt_time,
        prompt_ids.len()
    );

    let t_gen = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut next = argmax(&logits);
    for step in 0..n_gen {
        generated.push(next);
        if step + 1 < n_gen {
            let pos = prompt_ids.len() + step;
            logits = process_token(pos, next, &w, &mut cache_k, &mut cache_v);
            next = argmax(&logits);
        }
    }
    let gen_time = t_gen.elapsed();
    let matched = generated
        .iter()
        .zip(ref_gen.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "generated {} tokens in {:?} ({:.1} ms/token)",
        generated.len(),
        gen_time,
        gen_time.as_millis() as f64 / generated.len().max(1) as f64
    );
    println!(
        "greedy match vs numpy reference: first {matched} tokens identical (of {} compared)",
        ref_gen.len()
    );
    println!("GENERATED_IDS: {generated:?}");
}
